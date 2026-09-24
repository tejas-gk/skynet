export type Matrix = number[][];

export interface TransformerConfig { vocabSize: number; contextLength: number; embeddingSize: number; heads: number; layers: number; seed?: number; }
export interface ForwardResult { logits: Matrix; attentionWeights: number[][][][]; }
export interface GenerationOptions { maxNewTokens: number; temperature?: number; topK?: number; topP?: number; seed?: number; }
export interface AdamWState { step: number; first: number[]; second: number[]; }
export interface TransformerCheckpoint { config: TransformerConfig; vocabulary: string[]; weights: Matrix[]; optimizer?: AdamWState; }

export function softmax(values: readonly number[]): number[] {
  if (!values.length) return [];
  const maximum = Math.max(...values); const exponentials = values.map(value => Math.exp(value - maximum)); const total = exponentials.reduce((a, b) => a + b, 0);
  return exponentials.map(value => value / total);
}

class Scalar {
  grad = 0;
  constructor(public data: number, private readonly parents: Scalar[] = [], private readonly backwardFn: () => void = () => {}) {}
  add(other: Scalar): Scalar { const out = new Scalar(this.data + other.data, [this, other], () => { this.grad += out.grad; other.grad += out.grad; }); return out; }
  mul(other: Scalar): Scalar { const out = new Scalar(this.data * other.data, [this, other], () => { this.grad += other.data * out.grad; other.grad += this.data * out.grad; }); return out; }
  div(other: Scalar): Scalar { const out = new Scalar(this.data / other.data, [this, other], () => { this.grad += out.grad / other.data; other.grad -= this.data * out.grad / (other.data * other.data); }); return out; }
  neg(): Scalar { const out = new Scalar(-this.data, [this], () => { this.grad -= out.grad; }); return out; }
  exp(): Scalar { const value = Math.exp(this.data); const out = new Scalar(value, [this], () => { this.grad += value * out.grad; }); return out; }
  log(): Scalar { const out = new Scalar(Math.log(Math.max(this.data, 1e-12)), [this], () => { this.grad += out.grad / Math.max(this.data, 1e-12); }); return out; }
  sqrt(): Scalar { const value = Math.sqrt(this.data); const out = new Scalar(value, [this], () => { this.grad += out.grad / (2 * value); }); return out; }
  backward(): void { const order: Scalar[] = []; const visited = new Set<Scalar>(); const visit = (node: Scalar) => { if (visited.has(node)) return; visited.add(node); node.parents.forEach(visit); order.push(node); }; visit(this); this.grad = 1; for (let i = order.length - 1; i >= 0; i--) order[i].backwardFn(); }
}

type SMatrix = Scalar[][];
const scalarMatrix = (rows: number, columns: number, fill = 0): SMatrix => Array.from({ length: rows }, () => Array.from({ length: columns }, () => new Scalar(fill)));
const numeric = (values: SMatrix): Matrix => values.map(row => row.map(value => value.data));
const addMatrices = (a: SMatrix, b: SMatrix): SMatrix => a.map((row, i) => row.map((value, j) => value.add(b[i][j])));
const linear = (input: SMatrix, weights: SMatrix): SMatrix => input.map(row => weights[0].map((_, column) => row.reduce((sum, value, index) => sum.add(value.mul(weights[index][column])), new Scalar(0))));
const transpose = (input: SMatrix): SMatrix => input[0].map((_, column) => input.map(row => row[column]));

function norm(input: SMatrix): SMatrix {
  return input.map(row => {
    const mean = row.reduce((sum, value) => sum.add(value), new Scalar(0)).mul(new Scalar(1 / row.length));
    const variance = row.reduce((sum, value) => sum.add(value.add(mean.neg()).mul(value.add(mean.neg()))), new Scalar(0)).mul(new Scalar(1 / row.length));
    return row.map(value => value.add(mean.neg()).div(variance.add(new Scalar(1e-5)).sqrt()));
  });
}

function sample(probabilities: readonly number[], random: () => number): number { let target = random(); for (let i = 0; i < probabilities.length; i++) { target -= probabilities[i]; if (target <= 0) return i; } return probabilities.length - 1; }

export class Transformer {
  private readonly tokenEmbedding: SMatrix; private readonly positionEmbedding: SMatrix; private readonly query: SMatrix[]; private readonly key: SMatrix[]; private readonly value: SMatrix[]; private readonly attentionOutput: SMatrix[]; private readonly feedForwardIn: SMatrix[]; private readonly feedForwardOut: SMatrix[]; private readonly output: SMatrix; private readonly parameters: Scalar[] = [];
  private readonly headSize: number;

  constructor(readonly config: TransformerConfig, weights?: Matrix[]) {
    if (config.embeddingSize % config.heads !== 0) throw new Error("embeddingSize must be divisible by heads");
    if (config.vocabSize < 1 || config.contextLength < 1 || config.layers < 1 || config.heads < 1) throw new Error("Transformer dimensions must be positive");
    this.headSize = config.embeddingSize / config.heads; let state = (config.seed ?? 42) >>> 0;
    const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 0x100000000; }; const scale = 1 / Math.sqrt(config.embeddingSize); let cursor = 0;
    const parameterMatrix = (rows: number, columns: number): SMatrix => { const values = weights?.[cursor++]; const result = Array.from({ length: rows }, (_, i) => Array.from({ length: columns }, (_, j) => new Scalar(values?.[i]?.[j] ?? (random() * 2 - 1) * scale))); result.flat().forEach(value => this.parameters.push(value)); return result; };
    this.tokenEmbedding = parameterMatrix(config.vocabSize, config.embeddingSize); this.positionEmbedding = parameterMatrix(config.contextLength, config.embeddingSize); this.query = []; this.key = []; this.value = []; this.attentionOutput = []; this.feedForwardIn = []; this.feedForwardOut = [];
    for (let layer = 0; layer < config.layers; layer++) { this.query.push(parameterMatrix(config.embeddingSize, config.embeddingSize)); this.key.push(parameterMatrix(config.embeddingSize, config.embeddingSize)); this.value.push(parameterMatrix(config.embeddingSize, config.embeddingSize)); this.attentionOutput.push(parameterMatrix(config.embeddingSize, config.embeddingSize)); this.feedForwardIn.push(parameterMatrix(config.embeddingSize, config.embeddingSize * 4)); this.feedForwardOut.push(parameterMatrix(config.embeddingSize * 4, config.embeddingSize)); }
    this.output = parameterMatrix(config.embeddingSize, config.vocabSize);
  }

  get parameterCount(): number { return this.parameters.length; }

  private run(tokenIds: readonly number[]): { logits: SMatrix; attentionWeights: number[][][][] } {
    let hidden: SMatrix = tokenIds.map((id, position) => this.tokenEmbedding[id].map((value, dimension) => value.add(this.positionEmbedding[position][dimension]))); const attentionWeights: number[][][][] = [];
    for (let layer = 0; layer < this.config.layers; layer++) {
      const normalized = norm(hidden); const q = linear(normalized, this.query[layer]); const k = linear(normalized, this.key[layer]); const v = linear(normalized, this.value[layer]); const attended = scalarMatrix(tokenIds.length, this.config.embeddingSize); const layerWeights: number[][][] = [];
      for (let head = 0; head < this.config.heads; head++) { const weights: number[][] = []; for (let queryPosition = 0; queryPosition < tokenIds.length; queryPosition++) { const scores: Scalar[] = []; for (let keyPosition = 0; keyPosition < tokenIds.length; keyPosition++) { if (keyPosition > queryPosition) scores.push(new Scalar(Number.NEGATIVE_INFINITY)); else { let score = new Scalar(0); for (let d = 0; d < this.headSize; d++) score = score.add(q[queryPosition][head * this.headSize + d].mul(k[keyPosition][head * this.headSize + d])); scores.push(score.mul(new Scalar(1 / Math.sqrt(this.headSize)))); } } const probabilities = scores.map(score => score.exp()); const denominator = probabilities.reduce((sum, value) => sum.add(value), new Scalar(0)); const row = probabilities.map(value => value.div(denominator)); weights.push(row.map(value => value.data)); for (let d = 0; d < this.headSize; d++) for (let keyPosition = 0; keyPosition <= queryPosition; keyPosition++) attended[queryPosition][head * this.headSize + d] = attended[queryPosition][head * this.headSize + d].add(row[keyPosition].mul(v[keyPosition][head * this.headSize + d])); } layerWeights.push(weights); }
      attentionWeights.push(layerWeights); hidden = addMatrices(hidden, linear(attended, this.attentionOutput[layer])); const ff = linear(norm(hidden), this.feedForwardIn[layer]).map(row => row.map(value => value.data > 0 ? value : new Scalar(0))); hidden = addMatrices(hidden, linear(ff, this.feedForwardOut[layer]));
    }
    return { logits: hidden.map(row => linear([row], this.output)[0]), attentionWeights };
  }

  forward(tokenIds: readonly number[]): ForwardResult { this.validate(tokenIds); const result = this.run(tokenIds); return { logits: numeric(result.logits), attentionWeights: result.attentionWeights }; }

  trainStep(inputs: readonly number[], targets: readonly number[], learningRate: number, state: AdamWState, weightDecay = 0.01): number {
    this.validate(inputs); if (inputs.length !== targets.length || targets.some(id => id < 0 || id >= this.config.vocabSize)) throw new Error("inputs and targets must have equal valid lengths");
    this.parameters.forEach(parameter => { parameter.grad = 0; }); const result = this.run(inputs); let loss = new Scalar(0);
    for (let position = 0; position < targets.length; position++) { const probabilities = result.logits[position].map(value => value.exp()); const denominator = probabilities.reduce((sum, value) => sum.add(value), new Scalar(0)); loss = loss.add(probabilities[targets[position]].mul(new Scalar(1 / denominator.data)).log().neg()); }
    loss = loss.mul(new Scalar(1 / targets.length)); loss.backward(); state.step++; const beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
    for (let i = 0; i < this.parameters.length; i++) { const parameter = this.parameters[i]; state.first[i] = beta1 * state.first[i] + (1 - beta1) * parameter.grad; state.second[i] = beta2 * state.second[i] + (1 - beta2) * parameter.grad ** 2; const m = state.first[i] / (1 - beta1 ** state.step); const v = state.second[i] / (1 - beta2 ** state.step); parameter.data -= learningRate * (m / (Math.sqrt(v) + epsilon) + weightDecay * parameter.data); }
    return loss.data;
  }

  crossEntropy(inputs: readonly number[], targets: readonly number[]): number {
    this.validate(inputs); if (inputs.length !== targets.length) throw new Error("inputs and targets must have equal length");
    if (targets.some(id => !Number.isInteger(id) || id < 0 || id >= this.config.vocabSize)) throw new Error("targets contain an invalid token id");
    const logits = this.run(inputs).logits; let loss = 0;
    for (let position = 0; position < targets.length; position++) loss -= Math.log(Math.max(softmax(numeric([logits[position]])[0])[targets[position]], 1e-12));
    return loss / targets.length;
  }

  generate(prompt: readonly number[], options: GenerationOptions): number[] { if (options.maxNewTokens < 0 || !Number.isInteger(options.maxNewTokens)) throw new Error("maxNewTokens must be a non-negative integer"); const temperature = options.temperature ?? 1; if (!(temperature > 0)) throw new Error("temperature must be greater than zero"); const topK = Math.max(0, Math.floor(options.topK ?? 0)); const topP = options.topP ?? 1; if (topP <= 0 || topP > 1) throw new Error("topP must be between zero and one"); const result = [...prompt]; let state = (options.seed ?? 12345) >>> 0; const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 0x100000000; }; for (let step = 0; step < options.maxNewTokens; step++) { const logits = this.forward(result.slice(-this.config.contextLength)).logits.at(-1)!; const candidates = logits.map((logit, id) => ({ id, logit: logit / temperature })).sort((a, b) => b.logit - a.logit); const limited = topK ? candidates.slice(0, Math.min(topK, candidates.length)) : candidates; const probabilities = softmax(limited.map(candidate => candidate.logit)); let total = 0; let count = 0; for (; count < probabilities.length; count++) { total += probabilities[count]; if (total >= topP) { count++; break; } } const selected = limited.slice(0, Math.max(1, count)); result.push(selected[sample(softmax(selected.map(candidate => candidate.logit)), random)].id); } return result; }

  checkpoint(vocabulary: string[], optimizer?: AdamWState): TransformerCheckpoint { const matrices: Matrix[] = []; const collect = (matrix: SMatrix) => matrices.push(numeric(matrix)); collect(this.tokenEmbedding); collect(this.positionEmbedding); for (let layer = 0; layer < this.config.layers; layer++) { collect(this.query[layer]); collect(this.key[layer]); collect(this.value[layer]); collect(this.attentionOutput[layer]); collect(this.feedForwardIn[layer]); collect(this.feedForwardOut[layer]); } collect(this.output); return { config: this.config, vocabulary: [...vocabulary], weights: matrices, optimizer }; }
  private validate(tokenIds: readonly number[]): void { if (tokenIds.length === 0 || tokenIds.length > this.config.contextLength) throw new Error(`input length must be between 1 and ${this.config.contextLength}`); if (tokenIds.some(id => !Number.isInteger(id) || id < 0 || id >= this.config.vocabSize)) throw new Error("input contains an invalid token id"); }
}
