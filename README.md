# Skynet

An educational, Karpathy-style language-model data pipeline. The first slice
keeps the system deliberately small:

- `scraper/` is a bounded Go crawler for public, explicitly allowed domains.
- `src/` is a TypeScript dataset normalizer and inspector.
- `config.sprout` is the Sprout source of truth for corpus policy.

This does not scrape "all data". It requires a seed list, stays on the listed
domains, follows `robots.txt`, rate-limits requests, and stores provenance.
Only collect material you are allowed to use for training.

## Quick start

```sh
go run ./scraper -seeds https://www.gutenberg.org/files/1342/1342-0.txt,https://www.gutenberg.org/files/11/11-0.txt,https://www.gutenberg.org/files/1661/1661-0.txt,https://www.gutenberg.org/files/84/84-0.txt,https://www.gutenberg.org/files/2701/2701-0.txt -allow www.gutenberg.org -max-pages 100 -out data/raw.jsonl
npm install
npm run build
node dist/index.js normalize data/raw.jsonl data/train.jsonl
node dist/index.js stats data/train.jsonl
node dist/index.js train-tokenizer data/train.jsonl data/tokenizer.json --merges 512
node dist/index.js encode data/tokenizer.json "hello skynet"
node dist/index.js train data/train.jsonl data/checkpoint.json --steps 1000
node dist/index.js evaluate data/train.jsonl data/checkpoint.json
node dist/index.js inspect "hello sprout"
node dist/index.js chat data/train.jsonl data/checkpoint.json
```

The crawler defaults to 2 levels, 200 pages, and one request per second per
host. Use `-help` for the full set of limits.

## Data contract

Raw records contain `url`, `title`, `text`, `fetched_at`, and `content_hash`.
When a Gutenberg page links to a plain-text download, `source_url` records that
download while `url` remains the page where it was discovered.
The normalizer removes boilerplate whitespace, strips the Project Gutenberg
license header/footer, drops navigation and landing-page boilerplate, de-duplicates
by content hash, and writes the JSONL shape commonly consumed by a character or
BPE tokenizer:

```json
{"text":"...", "source":"https://example.com/article"}
```

The tokenizer and tiny Transformer trainer are kept separate from collection so
the corpus can be inspected before training.

`train-tokenizer` builds a byte-level BPE tokenizer with chat special tokens
`<|system|>`, `<|user|>`, `<|assistant|>`, and `<|endoftext|>`. The current
toy trainer still uses the character tokenizer; BPE is the next foundation for
the larger chat model.

## Model forward pass

`src/transformer.ts` implements a deterministic causal Transformer. `inspect`
prints per-position vocabulary logits, softmax
probabilities for the next token, and attention weights shaped as
`[layer][head][query][key]`. Training uses a scalar-autograd reference engine:
cross-entropy gradients flow through embeddings, causal attention, feed-forward
blocks, and layer normalization, and AdamW updates every parameter. The small
TypeScript reference model uses context length 32 to keep the graph usable;
larger models belong in the Go backend.

## Terminal Chat

```sh
node dist/index.js chat data/train.jsonl
```

The chat command reads a text file or normalized JSONL corpus and optionally a
checkpoint. It supports `/help`, `/temperature N`, `/top-k N`, `/top-p N`,
`/max-tokens N`, `/context`, `/reset`, and `/quit`. Chat keeps prior turns in
the prompt and truncates to the model context window, so each response is based
on recent conversation state.
Without a checkpoint it uses random weights; train first for useful output.

`train` saves all model matrices and AdamW state. `evaluate` reports
cross-entropy and perplexity from a checkpoint.
