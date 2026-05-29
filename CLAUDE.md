# MinSync

Manifest-based incremental vector DB indexing CLI tool written in Rust. No git required.

## Core Principles

- **Git-free**: manifest (mtime + size + content_hash) based change detection. Zero git dependency.
- **Rust native**: Pure Rust, no C++ dependencies. chonkie-core (chunk crate) built-in.
- **State directory**: `.minsync/` (config.toml, manifest.json, cursor.json, txn.json, lock)
- **Indexing target**: All files in directory. Exclude via `.minsyncignore` (`.gitignore` syntax).
- **Crash-safe**: Cursor updated only after all processing completes. mark+sweep guarantees convergence.
- **Deterministic ID**: `sha256(source_id + \0 + path + \0 + chunk_schema_id + \0 + chunk_type + \0 + content_hash + \0 + dup_index)`.

## Architecture

```
src/
├── main.rs           # CLI entry point (clap + tokio)
├── cli.rs            # clap derive definitions
├── lib.rs            # Module declarations
├── config.rs         # Config TOML parsing
├── error.rs          # thiserror error hierarchy, exit codes
├── manifest.rs       # ManifestBackend: scan, diff, atomic write
├── state.rs          # Cursor, Transaction, FileLock (fs2)
├── id.rs             # Deterministic doc_id + content_hash
├── normalize.rs      # Text normalization
├── types.rs          # Shared data models
├── sync.rs           # MinSync struct: init() + sync()
├── query.rs          # Query logic
├── verify.rs         # Verify, Check, Status
├── chunker/
│   ├── mod.rs        # Chunker trait
│   └── chonkie.rs    # chonkie-core based implementation
├── embedder/
│   ├── mod.rs        # Embedder trait (async)
│   └── openai.rs     # OpenAI HTTP backend
└── vectorstore/
    ├── mod.rs        # VectorStore trait + types
    ├── json_store.rs # JsonStore: cosine brute-force + JSON persist
    └── memory.rs     # InMemoryStore (testing only)
```

## CLI Commands

`init` / `sync` / `query` / `status` / `check` / `verify`

Exit codes: 0 (success), 1 (general error), 3 (Lock failed), 4 (VectorStore error), 5 (Embedding error)

## Key Dependencies

- `chunk` — chonkie-core SIMD chunker
- `clap` — CLI framework
- `tokio` + `reqwest` — async HTTP (OpenAI API)
- `ignore` — .minsyncignore processing (ripgrep engine)
- `sha2` — SHA-256 content hashing
- `tempfile` — atomic writes
- `fs2` — file locking
- `serde` + `serde_json` + `toml` — serialization

## Tests

103 tests (90 unit + 13 integration). Run with `cargo test`.
