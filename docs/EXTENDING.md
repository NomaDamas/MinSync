# Agent Extension Guide

This guide is for agents extending MinSync internals. Keep changes small and preserve the existing CLI contract: MinSync scans files with manifest-based change detection, chunks UTF-8 text, embeds changed chunks, writes vectors, then advances state only after the sync completes.

## Text Input Boundary

MinSync reads files with Rust UTF-8 text APIs. This is not a Markdown-only boundary. Any extension can be indexed when the file decodes as UTF-8.

Japanese, Chinese, Korean, and other Unicode text are handled as normal UTF-8 strings. Binary formats such as PDF, DOCX, XLSX, images, archives, and model files are not extracted. Agents should add those files to `.minsyncignore` unless another pipeline writes extracted text files for MinSync.

## Vector Store Backends

Current production backend: `lancedb`.

Test-only backend: `memory`.

To add a backend:

1. Add a module under `src/vectorstore/`.
2. Implement the `VectorStore` trait from `src/vectorstore/mod.rs`.
3. Preserve document metadata fields exactly: id, embedding, text, source_id, path, chunk_schema_id, chunk_type, heading_path, content_hash, and seen_token.
4. Implement the current filter subset: `Eq`, `Neq`, and `And`.
5. Validate vector dimensions before inserting or querying.
6. Keep cosine-compatible scoring or document the exact conversion to MinSync scores.
7. Wire the backend id into `create_vectorstore`.
8. Add tests for full sync, incremental add, modify, delete, query, metadata update, and stale sweep.

If the backend has asynchronous APIs, hide that behind a synchronous `VectorStore` facade as LanceDB does today, or refactor the trait deliberately with all callers updated in the same change.

## Embedding Providers

Current provider prefixes:

- `openai:` for OpenAI-compatible embeddings.
- `tei:` for Hugging Face Text Embeddings Inference.

Default model: `openai:text-embedding-3-small`.

To add a provider:

1. Add a module under `src/embedder/`.
2. Implement the `Embedder` trait from `src/embedder/mod.rs`.
3. Pick a stable id prefix in the form `provider:model`.
4. Wire the prefix into `create_embedder`.
5. Respect `batch_size`, `max_concurrent`, `timeout_seconds`, and `max_retries`.
6. Return exactly one vector per input text.
7. Override `embed_query` when the provider requires query-specific prefixes or endpoints.
8. Fail fast on malformed responses, auth failures, vector count mismatch, and invalid config.
9. Retry only transient failures such as timeouts, network errors, HTTP 429, and HTTP 5xx.
10. Document the model dimension and tell users to set `[vectorstore.options].dimension`.

Changing model dimension requires a full rebuild:

```bash
minsync sync --full
```

## Agent Checklist

Before handing off an extension:

1. Run `cargo fmt --all -- --check`.
2. Run `cargo clippy --all-targets --all-features -- -D warnings`.
3. Run `cargo test`.
4. Run one real CLI smoke test with `minsync init`, `minsync sync`, and `minsync query` when credentials or a local TEI server are available.
5. Update `skills/minsync/SKILL.md` when agent operating rules change.
