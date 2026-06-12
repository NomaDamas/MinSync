# MinSync

Manifest-based incremental vector DB indexing CLI tool written in Rust. No git required.

## Core Principles

- **Git-free**: manifest (mtime + size + content_hash) based change detection. Zero git dependency.
- **Rust native**: Pure Rust, no C++ dependencies. chonkie-core (chunk crate) built-in.
- **State directory**: `.minsync/` (config.toml, manifest.json, cursor.json, txn.json, lock)
- **Indexing target**: All files in directory. Exclude via `.minsyncignore` (`.gitignore` syntax).
- **Text files only**: MinSync reads files as UTF-8 text and does NO content extraction. Binary formats (PDF, DOCX, XLSX, images, ...) fail the UTF-8 read and are treated as empty — they are never chunked or embedded. Add them to `.minsyncignore`. Supporting them would require a separate text-extraction pipeline.
- **Crash-safe**: Cursor updated only after all processing completes. mark+sweep guarantees convergence.
- **Deterministic ID**: `sha256(source_id + \0 + path + \0 + chunk_schema_id + \0 + chunk_type + \0 + content_hash + \0 + dup_index)`.
