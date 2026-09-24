import { readFile } from "node:fs/promises";
import { CharTokenizer } from "./tokenizer.js";
import { AdamWState, Transformer, TransformerCheckpoint } from "./transformer.js";

export function adamWState(parameterCount: number): AdamWState { return { step: 0, first: Array(parameterCount).fill(0), second: Array(parameterCount).fill(0) }; }

export async function corpusFromJsonl(file: string): Promise<string> {
  const lines = (await readFile(file, "utf8")).split("\n"); let corpus = "";
  for (const line of lines) { if (!line.trim()) continue; const record = JSON.parse(line) as { text?: string }; if (record.text) corpus += `${record.text}\n`; }
  return corpus;
}

export function train(model: Transformer, tokens: readonly number[], steps: number, learningRate = 0.001): { loss: number; optimizer: AdamWState } {
  const state = adamWState(model.parameterCount); let loss = 0;
  if (tokens.length <= model.config.contextLength) throw new Error("training corpus must be longer than contextLength");
  for (let step = 0; step < steps; step++) {
    const start = (step * model.config.contextLength) % (tokens.length - model.config.contextLength - 1); const inputs = tokens.slice(start, start + model.config.contextLength); const targets = tokens.slice(start + 1, start + model.config.contextLength + 1);
    loss = model.trainStep(inputs, targets, learningRate, state); if ((step + 1) % Math.max(1, Math.floor(steps / 10)) === 0) console.log(`step ${step + 1}/${steps} loss ${loss.toFixed(4)}`);
  }
  return { loss, optimizer: state };
}

export function evaluate(model: Transformer, tokens: readonly number[]): { loss: number; perplexity: number } {
  const length = Math.min(tokens.length - 1, model.config.contextLength); if (length < 1) throw new Error("validation corpus is too short");
  const loss = model.crossEntropy(tokens.slice(0, length), tokens.slice(1, length + 1)); return { loss, perplexity: Math.exp(Math.min(loss, 50)) };
}

export function checkpointModel(model: Transformer, tokenizer: CharTokenizer, optimizer?: AdamWState): TransformerCheckpoint { return model.checkpoint(tokenizer.vocabulary, optimizer); }
