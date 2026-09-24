import test from "node:test";
import assert from "node:assert/strict";
import { BytePairTokenizer, CharTokenizer } from "./tokenizer.js";
import { softmax, Transformer } from "./transformer.js";
import { adamWState } from "./training.js";

test("character tokenizer round trips unicode text", () => {
  const tokenizer = new CharTokenizer("hello 🌱");
  assert.equal(tokenizer.decode(tokenizer.encode("🌱 hello")), "🌱 hello");
});

test("byte-pair tokenizer round trips arbitrary unicode text", () => {
  const tokenizer = BytePairTokenizer.train("hello hello 🌱 skynet", 16);
  const text = "hello 🌱 skynet\nこんにちは";
  assert.equal(tokenizer.decode(tokenizer.encode(text)), text);
});

test("byte-pair tokenizer learns merges and keeps chat special tokens atomic", () => {
  const tokenizer = BytePairTokenizer.train("banana banana banana", 8);
  const plainBytes = Buffer.from("banana banana", "utf8").length;
  const encoded = tokenizer.encode("<|user|>banana banana<|assistant|>");
  assert.ok(encoded.length < plainBytes + 2);
  assert.equal(tokenizer.decode(encoded), "<|user|>banana banana<|assistant|>");
  assert.equal(encoded[0], tokenizer.size - 2);
  assert.equal(encoded.at(-1), tokenizer.size - 1);
});

test("byte-pair tokenizer serializes merges and special tokens", () => {
  const tokenizer = BytePairTokenizer.train("the theater theme", 10);
  const restored = BytePairTokenizer.fromJSON(tokenizer.toJSON());
  const text = "<|system|>the theme";
  assert.deepEqual(restored.encode(text), tokenizer.encode(text));
  assert.equal(restored.decode(restored.encode(text)), text);
});

test("softmax is stable and normalized", () => {
  const probabilities = softmax([1000, 1001, 999]);
  assert.ok(probabilities.every(Number.isFinite));
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("transformer returns causal attention, logits, and normalized weights", () => {
  const tokenizer = new CharTokenizer("abcabc");
  const model = new Transformer({ vocabSize: tokenizer.size, contextLength: 8, embeddingSize: 16, heads: 4, layers: 2, seed: 1 });
  const result = model.forward(tokenizer.encode("abc"));
  assert.deepEqual(result.logits.map(row => row.length), [3, 3, 3]);
  assert.deepEqual([result.attentionWeights.length, result.attentionWeights[0].length, result.attentionWeights[0][0].length], [2, 4, 3]);
  for (const layer of result.attentionWeights) for (const head of layer) for (let query = 0; query < head.length; query++) {
    assert.ok(Math.abs(head[query].reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
    assert.ok(head[query].slice(query + 1).every(value => value === 0));
  }
});

test("generation respects context and is deterministic with a seed", () => {
  const model = new Transformer({ vocabSize: 3, contextLength: 4, embeddingSize: 8, heads: 2, layers: 1, seed: 2 });
  const first = model.generate([0, 1], { maxNewTokens: 5, temperature: 0.8, topK: 2, seed: 9 });
  const second = model.generate([0, 1], { maxNewTokens: 5, temperature: 0.8, topK: 2, seed: 9 });
  assert.deepEqual(first, second);
  assert.equal(first.length, 7);
});

test("cross-entropy training updates the AdamW vocabulary head", () => {
  const model = new Transformer({ vocabSize: 3, contextLength: 4, embeddingSize: 8, heads: 2, layers: 1, seed: 2 });
  const state = adamWState(model.parameterCount);
  const before = model.forward([0, 1, 2, 0]).logits[0][1];
  const loss = model.trainStep([0, 1, 2, 0], [1, 2, 0, 1], 0.01, state);
  assert.ok(Number.isFinite(loss));
  assert.notEqual(model.forward([0, 1, 2, 0]).logits[0][1], before);
  assert.equal(state.step, 1);
});
