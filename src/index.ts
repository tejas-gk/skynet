import { createReadStream, createWriteStream, readFileSync, writeFileSync } from "node:fs";
import { createInterface } from "node:readline";
import { BytePairTokenizer, type BytePairTokenizerState, CharTokenizer } from "./tokenizer.js";
import { softmax, Transformer, type TransformerCheckpoint } from "./transformer.js";
import { checkpointModel, corpusFromJsonl, evaluate, train } from "./training.js";

type Raw = { url: string; title?: string; text: string; content_hash?: string };
type Training = { text: string; source: string };

async function records(file: string, onRecord: (record: Raw) => void): Promise<void> {
  const input = createInterface({ input: createReadStream(file), crlfDelay: Infinity });
  for await (const line of input) { if (!line.trim()) continue; try { onRecord(JSON.parse(line) as Raw); } catch { console.error(`skipping invalid JSONL line`); } }
}

async function normalize(input: string, output: string): Promise<void> {
  const writer = createWriteStream(output); const hashes = new Set<string>(); let count = 0;
  await records(input, raw => {
    const text = stripGutenbergShell(raw.text).replace(/\s+/g, " ").trim();
    if (text.length < 80 || isLandingJunk(text) || isSignInJunk(text)) return;
    const key = raw.content_hash ?? text; if (hashes.has(key)) return; hashes.add(key); writer.write(`${JSON.stringify({ text, source: raw.url } satisfies Training)}\n`); count++;
  });
  await new Promise<void>((resolve, reject) => { writer.end(resolve); writer.on("error", reject); }); console.log(`wrote ${count} training records to ${output}`);
}

function stripGutenbergShell(value: string): string {
  let text = value;
  const start = text.match(/\*\*\*\s*START OF[^]*?PROJECT GUTENBERG[^]*?\*\*\*/i);
  if (start?.index !== undefined) text = text.slice(start.index + start[0].length);
  const end = text.search(/\*\*\*\s*END OF[^]*?PROJECT GUTENBERG[^]*?\*\*\*/i);
  if (end !== -1) text = text.slice(0, end);
  return text;
}

const LANDING_MARKERS = [
  "Project Gutenberg 79,270 free eBooks",
  "Reading Options & Kindle",
  "Frequently Downloaded",
  "Readers also downloaded",
  "Displaying results",
  "Sign in with Google",
  "About Project Gutenberg",
  "Main Categories",
  "Reading Lists",
  "Search Options",
  "Index of /files",
];

function isLandingJunk(text: string): boolean {
  return LANDING_MARKERS.some(marker => text.includes(marker)) && text.length < 40000;
}

function isSignInJunk(text: string): boolean {
  return text.includes("Sign in with Google") && text.includes("Forgot email");
}

async function stats(input: string): Promise<void> { let count = 0, chars = 0; await records(input, raw => { count++; chars += raw.text.length; }); console.log(JSON.stringify({ records: count, characters: chars, average_characters: count ? Math.round(chars / count) : 0 }, null, 2)); }

async function corpusText(input: string): Promise<string> { let text = ""; await records(input, raw => { text += `${raw.text}\n`; }); return text; }

async function trainModel(input: string, output: string, steps: number): Promise<void> {
  const corpus = input.endsWith(".jsonl") ? await corpusFromJsonl(input) : readFileSync(input, "utf8");
  const tokenizer = new CharTokenizer(corpus); const model = new Transformer({ vocabSize: tokenizer.size, contextLength: 32, embeddingSize: 32, heads: 4, layers: 2, seed: 7 }); const tokens = tokenizer.encode(corpus); const split = Math.floor(tokens.length * 0.9);
  const training = train(model, tokens.slice(0, split), steps); const validation = evaluate(model, tokens.slice(split)); writeFileSync(output, JSON.stringify(checkpointModel(model, tokenizer, training.optimizer), null, 2)); console.log(`saved checkpoint to ${output}; final loss ${training.loss.toFixed(4)} validation loss ${validation.loss.toFixed(4)} perplexity ${validation.perplexity.toFixed(2)}`);
}

async function trainTokenizer(input: string, output: string, merges: number): Promise<void> {
  const corpus = input.endsWith(".jsonl") ? await corpusFromJsonl(input) : readFileSync(input, "utf8");
  const tokenizer = BytePairTokenizer.train(corpus, merges);
  writeFileSync(output, JSON.stringify(tokenizer.toJSON(), null, 2));
  console.log(`saved BPE tokenizer with ${tokenizer.size} tokens and ${tokenizer.merges.length} merges to ${output}`);
}

function encodeWithTokenizer(tokenizerFile: string, text: string): void {
  const tokenizer = BytePairTokenizer.fromJSON(JSON.parse(readFileSync(tokenizerFile, "utf8")) as BytePairTokenizerState);
  const tokens = tokenizer.encode(text);
  console.log(JSON.stringify({ tokens, decoded: tokenizer.decode(tokens) }, null, 2));
}

async function evaluateModel(input: string, checkpointFile: string): Promise<void> {
  const corpus = input.endsWith(".jsonl") ? await corpusFromJsonl(input) : readFileSync(input, "utf8"); const checkpoint = JSON.parse(readFileSync(checkpointFile, "utf8")) as TransformerCheckpoint;
  const tokenizer = CharTokenizer.fromVocabulary(checkpoint.vocabulary); const model = new Transformer(checkpoint.config, checkpoint.weights); const tokens = tokenizer.encode(corpus); console.log(JSON.stringify(evaluate(model, tokens), null, 2));
}

async function chat(input: string, checkpointFile?: string): Promise<void> {
  const corpus = input.endsWith(".jsonl") ? await corpusText(input) : readFileSync(input, "utf8");
  const checkpoint = checkpointFile ? JSON.parse(readFileSync(checkpointFile, "utf8")) as TransformerCheckpoint : undefined;
  const tokenizer = checkpoint ? CharTokenizer.fromVocabulary(checkpoint.vocabulary) : new CharTokenizer(corpus);
  const model = new Transformer(checkpoint?.config ?? { vocabSize: tokenizer.size, contextLength: 128, embeddingSize: 64, heads: 4, layers: 2, seed: 7 }, checkpoint?.weights);
  const terminal = createInterface({ input: process.stdin, output: process.stdout });
  console.log(`Skynet terminal chat (${checkpoint ? "loaded checkpoint" : "untrained weights"}). Type /help or /quit.`);
  const activeContextLength = Math.min(model.config.contextLength, 32);
  let temperature = 0.8; let topK = 20; let topP = 1; let maxTokens = 20;
  let transcript = "The following is a conversation between a user and Skynet.\n";
  terminal.setPrompt("you> "); terminal.prompt();
  for await (const line of terminal) {
    if (line === "/quit" || line === "/exit") break;
    if (line === "/help") { console.log("/temperature N, /top-k N, /top-p N, /max-tokens N, /context, /reset, /quit"); terminal.prompt(); continue; }
    if (line === "/reset") { transcript = "The following is a conversation between a user and Skynet.\n"; console.log("context reset"); terminal.prompt(); continue; }
    if (line === "/context") { console.log(transcript.trim() || "context is empty"); terminal.prompt(); continue; }
    if (line.startsWith("/temperature ")) { temperature = Number(line.slice(13)); console.log(`temperature=${temperature}`); terminal.prompt(); continue; }
    if (line.startsWith("/top-k ")) { topK = Number(line.slice(7)); console.log(`top-k=${topK}`); terminal.prompt(); continue; }
    if (line.startsWith("/top-p ")) { topP = Number(line.slice(7)); console.log(`top-p=${topP}`); terminal.prompt(); continue; }
    if (line.startsWith("/max-tokens ")) { maxTokens = Math.min(40, Number(line.slice(12))); console.log(`max-tokens=${maxTokens}`); terminal.prompt(); continue; }
    if (!line.trim()) { terminal.prompt(); continue; }
    try {
      const promptText = `${transcript}User: ${line}\nSkynet:`;
      const prompt = encodeKnownCharacters(tokenizer, promptText).slice(-activeContextLength);
      if (!prompt.length) throw new Error("prompt has no characters from the model vocabulary");
      const generated = model.generate(prompt, { maxNewTokens: maxTokens, temperature, topK, topP });
      const response = tokenizer.decode(generated.slice(prompt.length)).split("\nUser:")[0].trimEnd();
      transcript += `User: ${line}\nSkynet: ${response}\n`;
      console.log(`skynet> ${response}`);
    }
    catch (error) { console.error(error instanceof Error ? error.message : error); }
    terminal.prompt();
  }
  terminal.close();
}

function encodeKnownCharacters(tokenizer: CharTokenizer, text: string): number[] {
  const tokens: number[] = [];
  for (const character of Array.from(text)) {
    try { tokens.push(...tokenizer.encode(character)); }
    catch { /* Character-level checkpoints cannot represent unseen characters. */ }
  }
  return tokens;
}

function inspect(text: string): void {
  const tokenizer = new CharTokenizer(text);
  const model = new Transformer({ vocabSize: tokenizer.size, contextLength: Math.max(8, text.length), embeddingSize: 32, heads: 4, layers: 2, seed: 7 });
  const result = model.forward(tokenizer.encode(text));
  const lastLogits = result.logits[result.logits.length - 1];
  console.log(JSON.stringify({ tokens: tokenizer.encode(text), vocabulary: tokenizer.vocabulary, logits: result.logits, nextTokenProbabilities: softmax(lastLogits), attentionWeights: result.attentionWeights }, null, 2));
}

const [command, input, output] = process.argv.slice(2);
if (command === "normalize" && input && output) await normalize(input, output);
else if (command === "stats" && input) await stats(input);
else if (command === "train" && input && output) await trainModel(input, output, Number(process.argv[5] === "--steps" ? process.argv[6] : 1000));
else if (command === "train-tokenizer" && input && output) await trainTokenizer(input, output, Number(process.argv[5] === "--merges" ? process.argv[6] : 512));
else if (command === "encode" && input && output) encodeWithTokenizer(input, output);
else if (command === "evaluate" && input && output) await evaluateModel(input, output);
else if (command === "inspect" && input) inspect(input);
else if (command === "chat" && input) await chat(input, output);
else { console.error("usage: node dist/index.js normalize <raw.jsonl> <train.jsonl> | stats <jsonl> | train-tokenizer <corpus> <tokenizer.json> [--merges N] | encode <tokenizer.json> <text> | train <corpus> <checkpoint.json> [--steps N] | evaluate <corpus> <checkpoint.json> | inspect <text> | chat <corpus.txt|train.jsonl> [checkpoint.json]"); process.exitCode = 1; }
