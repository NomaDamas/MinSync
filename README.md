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

### Vector store

MinSync stores vectors in an embedded [LanceDB](https://github.com/lancedb/lancedb) database (`vectorstore.id = "lancedb"`, the default). The LanceDB build vendors `protoc` automatically (needs a C compiler, standard with rustup).

Set the embedding dimension to match your embedder in `.minsync/config.toml`:

```toml
[vectorstore]
id = "lancedb"
[vectorstore.options]
dimension = 1536   # openai:text-embedding-3-small; use 384 for e5-small, 1024 for bge-m3
```

**ANN indexing.** A fresh table is searched by exhaustive flat scan (exact, 100% recall, but O(n)). Once a table grows past `index_build_threshold` chunks (default 256), MinSync builds an IVF-HNSW-SQ approximate-nearest-neighbour index — pinned to cosine distance to match query time — on the next `sync`/`flush`, so similarity search is accelerated. Newly synced chunks are immediately searchable via LanceDB's combined indexed + flat-over-delta search; once the unindexed delta reaches `index_optimize_delta_threshold` (default 10,000) MinSync folds it into the existing index incrementally (no full rebuild, no k-means retrain). No manual step is required.

These two thresholds are independent tuning knobs, **not** a min/max pair and **not** capacity limits — any number of vectors can be stored and searched. `index_build_threshold` gates the one-time index build (compared against total rows); `index_optimize_delta_threshold` gates each incremental optimize (compared against the unindexed delta only). Override them in `.minsync/config.toml`:

```toml
[vectorstore.options]
dimension = 1536
index_build_threshold = 256              # build the ANN index once total rows hit this
index_optimize_delta_threshold = 50000   # raise to optimize less often; lower for tighter query latency
```

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

`intfloat/multilingual-e5-small` produces 384-dimensional vectors. Set the dimension to match the embedder:

```toml
[vectorstore]
id = "lancedb"
[vectorstore.options]
dimension = 384
```

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
