# MinSync

[![Release](https://img.shields.io/github/v/release/vkehfdl1/MinSync)](https://img.shields.io/github/v/release/vkehfdl1/MinSync)
[![Build status](https://img.shields.io/github/actions/workflow/status/vkehfdl1/MinSync/main.yml?branch=main)](https://github.com/vkehfdl1/MinSync/actions/workflows/main.yml?query=branch%3Amain)
[![License](https://img.shields.io/github/license/vkehfdl1/MinSync)](https://img.shields.io/github/license/vkehfdl1/MinSync)

Git diff-based incremental vector index for any repository. No external DB required.

- **Github repository**: <https://github.com/NomaDamas/MinSync/>
- **Documentation**: <https://vkehfdl1.github.io/MinSync/>

## What it does

MinSync watches your git repository and keeps a local vector index in sync. It uses `git diff` to detect changes and only re-embeds what actually changed — no full re-indexing needed.

- Indexes all git-tracked files (respects `.gitignore` automatically)
- Incremental sync via `git diff` — only changed chunks get re-embedded
- Deterministic chunk IDs — same content always produces the same ID
- Crash-safe — interrupted syncs recover automatically
- Embedded vector DB (zvec) — no server, just a local file in `.minsync/`
- `.minsyncignore` for excluding files you don't want indexed

## Install

```bash
pip install minsync[zvec]
```

Or with uv:

```bash
uv add "minsync[zvec]"
```

For development (editable install):

```bash
uv add --editable "/path/to/MinSync[zvec]"
```

### Embedding model setup

OpenAI (default):

```bash
pip install langchain-openai
export OPENAI_API_KEY="sk-..."
```

HuggingFace (local, no API key):

```bash
pip install langchain-huggingface sentence-transformers
```

## Quick start

```bash
cd your-repo

# Initialize
minsync init
# or with HuggingFace:
minsync init --embedder "huggingface:sentence-transformers/all-MiniLM-L6-v2"

# Build initial index
minsync sync

# Search
minsync query "authentication flow" --k 5

# After pulling new changes
git pull
minsync sync          # only re-embeds changed files

# Check index health
minsync verify
```

## CLI commands

| Command | Description |
|---|---|
| `minsync init` | Initialize `.minsync/` in current git repo |
| `minsync sync` | Incremental sync (or `--full` for rebuild) |
| `minsync query "text"` | Semantic search over indexed content |
| `minsync status` | Show sync status (up-to-date / behind / interrupted) |
| `minsync check` | Verify environment and dependencies |
| `minsync verify` | Check index consistency (with `--fix` to repair) |

Use `minsync <command> --help` for full option details.

## Python API

```python
from minsync import MinSync
from pathlib import Path

ms = MinSync(repo_path=Path("/path/to/repo"))
ms.init()
ms.sync()
results = ms.query("search text", k=10)

# Custom components
ms = MinSync(
    repo_path=Path("/path/to/repo"),
    chunker=MyChunker(),        # implements Chunker protocol
    embedder=MyEmbedder(),      # implements Embedder protocol
    vector_store=MyStore(),     # implements VectorStore protocol
)
```

## .minsyncignore

Works like `.gitignore`. Add patterns for git-tracked files you don't want indexed:

```gitignore
# Build artifacts
dist/
blog/

# Attachments
attachments/
*.png
*.pdf

# Config files
pyproject.toml
uv.lock
```

## How it works

1. `git diff` detects changed files since last sync
2. Changed files are re-chunked (markdown heading-based by default)
3. Each chunk gets a deterministic ID from `sha256(repo_id + path + content_hash + ...)`
4. Only chunks with new IDs get embedded — unchanged chunks skip the API call
5. Stale chunks (old content) are automatically swept

All state lives in `.minsync/` — delete it to start fresh.

## Development

```bash
git clone https://github.com/NomaDamas/MinSync.git
cd MinSync
make install
uv run pytest
```
