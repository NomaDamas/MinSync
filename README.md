# MinSync

Manifest-based incremental vector DB indexing CLI. No git required. Written in Rust.

## Install

```bash
git clone https://github.com/NomaDamas/MinSync.git
cd MinSync
cargo build --release
export OPENAI_API_KEY="sk-..."
```

## Usage

```bash
minsync init                          # initialize .minsync/
minsync sync                          # index files (incremental)
minsync sync --full                   # rebuild from scratch
minsync query "search text" --k 5     # semantic search
minsync watch                         # watch .md/.txt files and incrementally re-index on change
minsync status                        # sync state
minsync check                         # health check
minsync verify --fix                  # consistency check + repair
```

## How it works

MinSync scans your directory, detects file changes via manifest comparison (mtime + size + SHA-256 content hash), chunks changed files with a recursive chunker (built on [chonkie-core](https://github.com/chonkie-inc/chunk)'s split/merge primitives) — paragraph→sentence→line boundaries merged to a size budget — embeds them via OpenAI, and stores vectors locally. Only changed content gets re-embedded. Stale chunks are automatically swept.

State lives in `.minsync/`. Delete it to start fresh.

### Vector stores

- **`json`** (default): local JSON file, brute-force cosine similarity.
- **`lancedb`** (optional): embedded LanceDB. Set `vectorstore.id = "lancedb"` and `[vectorstore.options] dimension = 1536` in `.minsync/config.toml`. The LanceDB build vendors `protoc` automatically (needs a C compiler, standard with rustup).

## Local embeddings (no OpenAI) — Hugging Face TEI

You can run MinSync entirely offline using a local [Text Embeddings Inference](https://github.com/huggingface/text-embeddings-inference) server. No `OPENAI_API_KEY` needed.

### 1. Install and launch TEI (macOS Apple Silicon)

```bash
brew install text-embeddings-inference
```

```bash
text-embeddings-router --model-id intfloat/multilingual-e5-small --port 8080 --dtype float32
```

The first run downloads the model (~470 MB) to `~/.cache/huggingface`. Once it's ready:

```bash
curl http://localhost:8080/health   # should return 200
```

### 2. Configure MinSync

Either run `minsync init --embedder tei:intfloat/multilingual-e5-small` and then edit `.minsync/config.toml`, or set the `[embedder]` section directly:

```toml
[embedder]
id = "tei:intfloat/multilingual-e5-small"
base_url = "http://localhost:8080"
query_prefix = "query: "
passage_prefix = "passage: "
```

`e5` models require input prefixes for best retrieval quality: `query: ` is prepended to search queries, `passage: ` to indexed documents. MinSync applies these automatically from the config.

### 3. LanceDB dimension note

`intfloat/multilingual-e5-small` produces 384-dimensional vectors. If you're using the LanceDB vector store, set the dimension explicitly:

```toml
[vectorstore]
id = "lancedb"
[vectorstore.options]
dimension = 384
```

The default `json` store doesn't require a dimension setting.

### 4. Run normally

```bash
minsync sync --full          # first sync (--full because init baselines the manifest)
minsync query "검색어" --k 5
minsync watch
```

### Alternative models

| Model | Dim | Notes |
|---|---|---|
| `intfloat/multilingual-e5-small` | 384 | Multilingual incl. Korean; recommended default |
| `dragonkue/multilingual-e5-small-ko-v2` | 384 | Korean-tuned variant of e5-small |
| `BAAI/bge-m3` | 1024 | No prefix needed; larger model |

## .minsyncignore

`.gitignore` syntax. Exclude files from indexing:

```gitignore
target/
*.png
*.pdf
```

## Development

```bash
cargo test            # 140 tests
cargo clippy          # lint
cargo fmt             # format
```

## License

MIT
