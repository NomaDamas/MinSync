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
minsync status                        # sync state
minsync check                         # health check
minsync verify --fix                  # consistency check + repair
```

## How it works

MinSync scans your directory, detects file changes via manifest comparison (mtime + size + SHA-256 content hash), chunks changed files with [chonkie-core](https://github.com/chonkie-inc/chunk), embeds them via OpenAI, and stores vectors locally. Only changed content gets re-embedded. Stale chunks are automatically swept.

State lives in `.minsync/`. Delete it to start fresh.

## .minsyncignore

`.gitignore` syntax. Exclude files from indexing:

```gitignore
target/
*.png
*.pdf
```

## Development

```bash
cargo test            # 103 tests
cargo clippy          # lint
cargo fmt             # format
```

## License

MIT
