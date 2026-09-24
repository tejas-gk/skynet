export class CharTokenizer {
  readonly vocabulary: string[];
  private readonly ids = new Map<string, number>();

  constructor(corpus: string) {
    this.vocabulary = [...new Set(Array.from(corpus))].sort((a, b) => a.codePointAt(0)! - b.codePointAt(0)!);
    if (this.vocabulary.length === 0) throw new Error("cannot build a tokenizer from an empty corpus");
    this.vocabulary.forEach((token, id) => this.ids.set(token, id));
  }

  static fromVocabulary(vocabulary: readonly string[]): CharTokenizer {
    const tokenizer = new CharTokenizer(vocabulary.join(""));
    if (tokenizer.vocabulary.length !== vocabulary.length || tokenizer.vocabulary.some((token, index) => token !== vocabulary[index])) {
      throw new Error("tokenizer vocabulary must contain unique characters in code-point order");
    }
    return tokenizer;
  }

  get size(): number { return this.vocabulary.length; }

  encode(text: string): number[] {
    return Array.from(text).map(token => {
      const id = this.ids.get(token);
      if (id === undefined) throw new Error(`token is not in vocabulary: ${JSON.stringify(token)}`);
      return id;
    });
  }

  decode(ids: readonly number[]): string {
    return ids.map(id => {
      const token = this.vocabulary[id];
      if (token === undefined) throw new Error(`token id is out of range: ${id}`);
      return token;
    }).join("");
  }
}

export interface BytePairTokenizerState { merges: [number, number][]; specialTokens?: string[]; }

const DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>"];

export class BytePairTokenizer {
  readonly merges: [number, number][];
  readonly specialTokens: string[];
  private readonly mergeRanks = new Map<string, number>();
  private readonly mergeIds = new Map<string, number>();
  private readonly tokenBytes = new Map<number, number[]>();
  private readonly specialIds = new Map<string, number>();
  private readonly idsToSpecial = new Map<number, string>();

  constructor(state: BytePairTokenizerState = { merges: [], specialTokens: DEFAULT_SPECIAL_TOKENS }) {
    this.merges = state.merges.map(([left, right]) => [left, right]);
    this.specialTokens = [...(state.specialTokens ?? DEFAULT_SPECIAL_TOKENS)];
    for (let byte = 0; byte < 256; byte++) this.tokenBytes.set(byte, [byte]);
    this.merges.forEach(([left, right], rank) => {
      this.validateTokenId(left); this.validateTokenId(right);
      const id = 256 + rank;
      this.mergeRanks.set(pairKey(left, right), rank);
      this.mergeIds.set(pairKey(left, right), id);
      this.tokenBytes.set(id, [...this.tokenBytes.get(left)!, ...this.tokenBytes.get(right)!]);
    });
    this.specialTokens.forEach((token, offset) => {
      const id = 256 + this.merges.length + offset;
      this.specialIds.set(token, id);
      this.idsToSpecial.set(id, token);
    });
  }

  static train(corpus: string, mergeCount: number, specialTokens = DEFAULT_SPECIAL_TOKENS): BytePairTokenizer {
    if (mergeCount < 0 || !Number.isInteger(mergeCount)) throw new Error("mergeCount must be a non-negative integer");
    let tokens = Array.from(Buffer.from(corpus, "utf8"));
    const merges: [number, number][] = [];
    for (let step = 0; step < mergeCount; step++) {
      const pairCounts = new Map<string, { left: number; right: number; count: number }>();
      for (let index = 0; index < tokens.length - 1; index++) {
        const left = tokens[index], right = tokens[index + 1], key = pairKey(left, right);
        const pair = pairCounts.get(key);
        if (pair) pair.count++;
        else pairCounts.set(key, { left, right, count: 1 });
      }
      const best = [...pairCounts.values()].sort((a, b) => b.count - a.count || a.left - b.left || a.right - b.right)[0];
      if (!best || best.count < 2) break;
      const nextId = 256 + merges.length;
      merges.push([best.left, best.right]);
      tokens = mergePair(tokens, best.left, best.right, nextId);
    }
    return new BytePairTokenizer({ merges, specialTokens });
  }

  static fromJSON(state: BytePairTokenizerState): BytePairTokenizer { return new BytePairTokenizer(state); }

  get size(): number { return 256 + this.merges.length + this.specialTokens.length; }

  toJSON(): BytePairTokenizerState { return { merges: this.merges.map(([left, right]) => [left, right]), specialTokens: [...this.specialTokens] }; }

  encode(text: string): number[] {
    const result: number[] = [];
    let cursor = 0;
    while (cursor < text.length) {
      const special = this.specialTokens.find(token => text.startsWith(token, cursor));
      if (special) { result.push(this.specialIds.get(special)!); cursor += special.length; continue; }
      let nextSpecial = text.length;
      for (const token of this.specialTokens) {
        const index = text.indexOf(token, cursor + 1);
        if (index !== -1) nextSpecial = Math.min(nextSpecial, index);
      }
      result.push(...this.encodeBytes(Array.from(Buffer.from(text.slice(cursor, nextSpecial), "utf8"))));
      cursor = nextSpecial;
    }
    return result;
  }

  decode(ids: readonly number[]): string {
    const chunks: string[] = [];
    let bytes: number[] = [];
    const flush = () => { if (bytes.length) { chunks.push(Buffer.from(bytes).toString("utf8")); bytes = []; } };
    for (const id of ids) {
      const special = this.idsToSpecial.get(id);
      if (special !== undefined) { flush(); chunks.push(special); continue; }
      const token = this.tokenBytes.get(id);
      if (!token) throw new Error(`token id is out of range: ${id}`);
      bytes.push(...token);
    }
    flush();
    return chunks.join("");
  }

  private encodeBytes(bytes: number[]): number[] {
    let ids = [...bytes];
    while (ids.length > 1) {
      let bestRank = Infinity, bestLeft = -1, bestRight = -1;
      for (let index = 0; index < ids.length - 1; index++) {
        const rank = this.mergeRanks.get(pairKey(ids[index], ids[index + 1]));
        if (rank !== undefined && rank < bestRank) { bestRank = rank; bestLeft = ids[index]; bestRight = ids[index + 1]; }
      }
      if (bestRank === Infinity) break;
      ids = mergePair(ids, bestLeft, bestRight, this.mergeIds.get(pairKey(bestLeft, bestRight))!);
    }
    return ids;
  }

  private validateTokenId(id: number): void {
    if (!this.tokenBytes.has(id)) throw new Error(`merge references unknown token id: ${id}`);
  }
}

function pairKey(left: number, right: number): string { return `${left},${right}`; }

function mergePair(tokens: readonly number[], left: number, right: number, merged: number): number[] {
  const result: number[] = [];
  for (let index = 0; index < tokens.length; index++) {
    if (tokens[index] === left && tokens[index + 1] === right) { result.push(merged); index++; }
    else result.push(tokens[index]);
  }
  return result;
}
