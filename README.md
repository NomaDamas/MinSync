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
