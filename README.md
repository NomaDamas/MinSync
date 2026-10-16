# MinSync

[English](README.md) | [한국어](README.ko.md)

![MinSync turns changed files into searchable vector chunks](assets/minsync-flow.svg)

MinSync is a manifest-based incremental vector database indexing CLI for text files. It does not need git: it tracks `mtime`, file size, and SHA-256 content hashes in `.minsync/`, re-embeds only changed chunks, sweeps stale chunks, and keeps a local LanceDB index ready for semantic search.

## Why MinSync

- **Git-free change detection**: works in any directory, including generated workspaces and agent sandboxes.
- **Incremental embeddings**: unchanged text is not chunked or embedded again.
- **Crash-safe state**: cursor updates happen only after processing completes.
- **Deterministic chunk IDs**: IDs derive from source, path, schema, content hash, and duplicate index.
- **Rust native**: a single CLI with chonkie-core chunking built in.
- **Text-only by design**: PDF, DOCX, XLSX, images, and other binary formats are treated as empty because MinSync does no extraction. Add them to `.minsyncignore`.

## Contents

- [Install](#install)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Chunkers](#chunkers)
- [Vector store](#vector-store)
- [Embedding network reliability](#embedding-network-reliability)
- [Local embeddings](#local-embeddings)
- [.minsyncignore](#minsyncignore)
- [Development](#development)

## Install

Recommended install path:

```bash
curl -fsSL https://raw.githubusercontent.com/NomaDamas/MinSync/main/scripts/install.sh | sh
```

The installer asks whether to star the repository with `gh repo star NomaDamas/MinSync`. If `gh` is unavailable or not authenticated, it continues without failing.

Direct Cargo install:

```bash
cargo install minsync
```

From source:

```bash
git clone https://github.com/NomaDamas/MinSync.git
cd MinSync
cargo build --release
```

MinSync embeds in-process with [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) by default. No API key, TEI server, or Hugging Face gate is required. The first `minsync sync` downloads the model into the local cache.

For OpenAI embeddings instead:

```bash
export OPENAI_API_KEY="sk-..."
minsync init --embedder openai:text-embedding-3-small
```

## Quick Start

```bash
minsync init                          # initialize .minsync/
minsync sync                          # index changed files incrementally
minsync sync --full                   # rebuild from scratch
minsync query "search text" --mode vector --k 5
minsync query "exact terms" --mode bm25 --k 5
minsync query "search text" --mode hybrid --k 5
minsync watch                         # initial sync, then re-index on file changes
minsync watch --watch-on-sync-error   # stay alive and retry after initial sync errors
minsync status                        # sync state
minsync check                         # health check
minsync verify --fix                  # consistency check + repair
minsync optimize                      # compact + re-index + prune the store
```

Select the BM25 tokenizer preset during initialization:

```bash
minsync init --language ko
```

Supported values are `simple`, `ko`, `ja`, `zh`, `ar`, and `multilingual`.
Korean uses Kiwi through the `kiwi-rs` Rust binding, Japanese uses an embedded
Lindera dictionary, Chinese uses `jieba-rs`, and Arabic uses an in-process light
stemmer based on Discrawl PR #180.
Changing `[lexical].language` triggers a full rebuild on the next sync.

`minsync watch` performs an initial sync before waiting for filesystem events.
By default, an initial embedding or vector-store error stops the command,
matching `minsync sync` fail-fast behavior. Use
`--watch-on-sync-error` to keep the watcher alive and retry on later file
events; failures remain visible in the log and the cursor advances only after
a later sync succeeds.

Sync output reports file-level changes separately from chunk-level storage
effects: `files added/modified/deleted` describes source files, while
`chunks inserted/reused/removed` describes content-addressed index rows.
JSON sync output also reports `files_checked`, `elapsed_seconds`,
`freshness_check_only`, and `query_ready`. An unchanged incremental sync
uses a conservative metadata fingerprint to skip hashing unchanged files,
then returns without reading, chunking, embedding, writing, flushing, or
rebuilding index rows; its elapsed time therefore measures freshness-check
cost. Changed and
full syncs include indexing and LanceDB maintenance in their elapsed time.
For a local, credential-free comparison, run `cargo run --quiet -- sync
--format json` twice, edit one tracked text file and run it again, then run
`cargo run --quiet -- sync --full --format json`. Query process startup is
outside this sync measurement.

Run the checked-in credential-free scale benchmark with:

```bash
cargo test --test issue_41_benchmark -- --nocapture
```

It generates the same local corpus at 1x, 2x, 4x, and 8x, reports full,
unchanged, and changed-file sync milliseconds, and reports p50/p95 for BM25,
vector, and hybrid retrieval. The benchmark intentionally uses the in-memory
store, so its query timings measure retrieval scaling separately from LanceDB
index maintenance; the LanceDB ANN/FTS threshold and unindexed-delta behavior
are covered by the live index tests. Timing values are observations, not a
machine-specific latency SLO.

Korean tokenization requires the official Kiwi native library and base model.
Set `KIWI_LIBRARY_PATH` to `libkiwi` and `KIWI_MODEL_PATH` to the extracted
`models/cong/base` directory. MinSync pins the unreleased ABI fix from
`JAICHANGPARK/kiwi-rs` until a corrected crates.io release is available.
On macOS/Linux, `bash scripts/install-kiwi.sh` downloads the matching official
assets and prints the required environment variables.
On Windows, run `powershell -ExecutionPolicy Bypass -File
scripts/install-kiwi.ps1` and set the printed paths in the environment.
The Windows CI uses the official Kiwi 0.22.2 assets until the remaining
`kiwi_config_t` ABI differences in the Rust binding are resolved upstream.

## How It Works

MinSync scans your directory, compares each text file to the manifest, chunks changed content once, and stores the same stable chunk IDs for vector and BM25 retrieval in LanceDB. Stale rows are removed by mark-and-sweep. Vector mode embeds the query, BM25 mode uses LanceDB full-text search without an embedding request, and hybrid mode combines both rankings with deterministic reciprocal rank fusion (RRF, `k=60`).

The selected analyzer converts documents and queries into the shared
`lexical_text` column before LanceDB BM25 indexing.

State lives in `.minsync/`:

| File | Purpose |
|---|---|
| `config.toml` | collection, chunker, embedder, vector store, normalization |
| `manifest.json` | last known file metadata and content hashes |
| `cursor.json` | last completed processing point |
| `txn.json` | in-progress transaction marker |
| `progress.jsonl` | files already committed by an interrupted sync (resume log) |
| `lock` | process lock |

Delete `.minsync/` to start fresh.

### Freshness checks

After the first completed sync, MinSync uses the persisted file metadata
fingerprint as a conservative fast path. On Unix this fingerprint includes
size, modification time, change time, device, and inode, so unchanged files
reuse their manifest hash without reading their contents. A replacement that
keeps the same size and restores the modification time still changes the
fingerprint through the filesystem change time and is rehashed.

Manifests written by older versions, platforms without the required metadata,
and explicit `minsync sync --full` scans conservatively hash file contents.
The fast path is not a second freshness owner: the manifest remains the only
source of content hashes, and a fingerprint mismatch always falls back to
SHA-256. Filesystem writers that mutate content after metadata is observed
remain subject to the normal concurrent-write race; run `--full` when an
authoritative rescan is required.

### Unreadable entries

A directory or file the scan cannot read
(`EACCES`, `EPERM`, `EIO`, `EBUSY`, `ETIMEDOUT`, `ESTALE`, symlink loops, ...)
no longer aborts the scan: it is skipped and reported in the sync summary's
`scan_errors` list (`path`, `kind` = `transient` or `permanent`, `error`) with
one warning per entry. A path that vanished mid-scan (`ENOENT`, `ENOTDIR`) is
treated as deleted. `status` and `verify` apply the same rule, so
`verify --fix` never removes chunks of an unreadable path.

With `[sync] on_file_error = "skip"` (default) each unreadable entry is also
listed in `files_failed` (stage `read`), so the run is `partial` and exits with
code 6; with `"fail"` the sync aborts at the first one, naming its path.

**Guarantee:** an entry that errored is *unknown*, never *deleted*. Its
previously indexed chunks are kept, including during `--full` sweeps, until a
later scan can read the path again (or confirms it is gone). If the root
directory itself is unreadable the sync still fails, since nothing can be said
about any file.

## Chunkers

| id | Strategy | Boundary stability under edits |
|---|---|---|
| `recursive` (default) | paragraph to sentence to line splits merged to a size budget | an edit near the top of a file can shift downstream boundaries |
| `chonkie` | delimiter/size-based chonkie-core chunking | same drift caveat as `recursive` |
| `cdc` | content-defined chunking with FastCDC-style rolling hash | small edits usually affect only nearby chunks |

The `recursive` chunker splits on paragraph breaks, sentence delimiters
(`. `, `? `, `! `), and line breaks, then merges those pieces up to
`chunker.options.max_chunk_size`. The configured size is a hard cap for
ordinary delimiter-based text and for delimiter-less lines, which are split
at character boundaries with a warning that includes the original length.

Contiguous Markdown table lines (optional leading whitespace, beginning with
`|` and containing another `|`) are treated as one table block. A valid table
header and separator stay together with data rows, and table chunks are
split only between rows. When a table needs multiple chunks, every chunk
after the first repeats the header and separator rows. If preserving a table
row or its repeated header would exceed the cap, MinSync keeps that table
structure intact and emits a warning; this is the only intentional exception
to the hard cap. A warning is also emitted when a table boundary is selected.

Select at init time:

```bash
minsync init --chunker cdc
```

Or edit `.minsync/config.toml`:

```toml
[chunker]
id = "cdc"
```

Switching the chunker changes the chunk schema, so run `minsync sync --full` afterwards.

## Vector Store

MinSync stores vectors in embedded LanceDB by default:

```toml
[vectorstore]
id = "lancedb"

[vectorstore.options]
dimension = 1024
index_build_threshold = 256
index_optimize_delta_threshold = 10000
```

`minsync init` sets `dimension` automatically for known embedders. For other models, set `dimension` to match your embedder:

| Embedder | Dimension |
|---|---:|
| `native:Qwen/Qwen3-Embedding-0.6B` (default) | 1024 |
| `tei:google/embeddinggemma-300m` | 768 |
| `openai:text-embedding-3-small` | 1536 |
| `tei:intfloat/multilingual-e5-small` | 384 |
| `tei:BAAI/bge-m3` | 1024 |

MinSync builds an IVF-HNSW-SQ ANN index after `index_build_threshold` rows and incrementally optimizes unindexed deltas after `index_optimize_delta_threshold` rows. These thresholds are tuning knobs, not capacity limits.

Agents adding another vector store should implement the `VectorStore` trait and wire the new id through `create_vectorstore`. See `docs/EXTENDING.md` and `skills/minsync/SKILL.md` for the agent-facing extension checklist.

### Write path, checkpoints and `optimize`

A sync writes files in batches of `[sync] batch_files` (default 25). Per batch it does one id-only existence check, one embedding call, one metadata update for reused chunks, one append for new chunks and one stale-chunk sweep, so the commit count is independent of how many files are in the batch. It keeps BTree indexes on `id` and `path` so these filters stay index lookups, and every `[sync] checkpoint_every_files` files (default 50) compacts fragments (once there are at least `compact_fragment_threshold`, default 64) and prunes old versions. A running sync logs `sync progress` at each checkpoint. A crash redoes at most one batch.

```toml
[sync]
batch_files = 25              # files per store commit; 1 = per-file commits
checkpoint_every_files = 50   # 0 disables mid-sync checkpoints

[vectorstore.options]
compact_fragment_threshold = 64
```

`minsync optimize` runs only compact + re-index + prune (no embedding, no file reads) to repair a store that earlier interrupted syncs left fragmented.

### Machine-readable progress

A host app that spawns `minsync sync` can read progress as JSON lines instead of parsing human logs:

```bash
minsync sync --progress jsonl                       # events on stdout, human logs on stderr
minsync --format json sync --progress-file /tmp/p.jsonl  # events to a file, result stays on stdout
minsync sync --progress jsonl --progress-heartbeat-seconds 2
```

```jsonc
{"event":"start","total_files":12345,"resumed":42,"full":false,"dry_run":false,"sync_token":"…"}
{"event":"batch","processed":125,"total":12345,"path_first":"a/b.md","path_last":"a/c.md","elapsed_ms":3200}
{"event":"file_failed","path":"notes/broken.pdf","stage":"read","error":"io error: Permission denied (os error 13)"}
{"event":"heartbeat","processed":125,"total":12345,"elapsed_ms":8200}
{"event":"done","processed":12345,"total":12345,"files_processed":12303,"files_added":12000,"files_modified":303,"files_deleted":0,"files_resumed":42,"chunks_added":…,"files_failed":1,"partial":true,"elapsed_seconds":1234.5}
```

- `start` precedes the first batch; one `batch` per committed batch (`processed` includes resumed files); one `file_failed` per skipped file (see [Skipped files](#skipped-files)), as soon as it is known; `done` is the last event and carries `files_failed` and `partial`.
- `heartbeat` is sent only after `--progress-heartbeat-seconds` (default 5) without another event, from a dedicated thread, so a slow embedding call or store commit still produces a signal. A growing `processed` means slow; an unchanged one across heartbeats means stuck on the current batch.
- Dry runs and up-to-date runs emit `start` (`sync_token: null`) and `done` only.
- With `--progress jsonl`, the text summary moves to stderr and `--format json` prints the final result as one compact last line on stdout. Without the flags, output is unchanged.
- A sink that closes mid-run (host exited, disk full) is dropped with a warning and never fails the sync. An unopenable `--progress-file` fails before any work. A `--progress-file` inside `.minsync/` (it would overwrite state such as the resume log) or inside the indexed workspace without a matching `.minsyncignore` entry (it would be indexed as a source file) is rejected before the file is touched.

## Skipped files

By default one bad file does not abort a sync. A file whose read, chunk, token-count or embed step fails (EACCES, EBUSY, EIO, a parser error, an embedder rejecting its text) is skipped and recorded; if a batch commit still fails, MinSync retries that batch file by file and skips only the culprit. A skipped file is never recorded as done: a new file stays out of the saved manifest and a previously indexed file keeps its previous manifest entry, so the next `minsync sync` retries it automatically. Its old chunks (if it was indexed before) stay searchable until it succeeds, including across full syncs.

```json
"files_failed": [
  { "path": "notes/broken.pdf", "stage": "read", "error": "io error: notes/broken.pdf: Permission denied (os error 13)" }
],
"partial": true
```

`stage` is one of `read`, `chunk`, `truncate`, `embed`, `store`. When `partial` is true the CLI exits with code **6** (completed, files skipped) and text output lists each skipped path on stderr. Hard failures keep their codes (1, 3, 4, 5). An embed/store failure that hits the first few files of a run with nothing indexed yet (for example the embedder being down) is treated as an outage, not as bad files: the sync aborts with that error (exit 5/4) and the file path in the message.

```toml
[sync]
on_file_error = "skip"   # "fail" aborts on the first failing file (the pre-0.5 behavior)
```

## Embedding Reliability

Embedding requests use per-request timeouts and retry transient failures: network errors, timeouts, HTTP 429, and HTTP 5xx. Permanent failures such as invalid auth, malformed responses, or validation errors fail immediately. A failed or killed sync never advances the cursor or manifest, but it keeps a resume log (`progress.jsonl`): the next `minsync sync` skips every file the interrupted run already committed and continues from there. The log is discarded if the files, chunker, normalizer, embedder, or lexical language changed in between.

```toml
[embedder]
max_retries = 3
timeout_seconds = 60
max_concurrent = 1
```

`timeout_seconds` also bounds the native (in-process) embedder: each forward pass (one batch) must finish within `per_batch_timeout_seconds` (default: `timeout_seconds`); `0` is rejected. A blocked native kernel cannot be cancelled, so on expiry `minsync sync` aborts with exit code `7` (`embedder timeout`) instead of hanging; the stuck thread dies with the process, and once it happens every later native call in that process fails fast. `progress.jsonl` keeps every file committed before the timeout, so re-running `minsync sync` resumes without re-embedding them. `minsync check` forward passes are bounded the same way. Model download/load is not bounded. On CPU, large batches can legitimately take long; raise `per_batch_timeout_seconds` there.

Each native forward pass is also planned to stay within `max_batch_tokens` padded tokens (inputs × longest input, default 4096) on top of `batch_size` and an attention-memory cap, so a pass of long chunks finishes in seconds instead of approaching the deadline. An input longer than the cap is still embedded alone.

```toml
[embedder]
timeout_seconds = 60
per_batch_timeout_seconds = 120   # optional, native only
max_batch_tokens = 4096           # optional, native only
```

`base_url` also works for OpenAI-compatible gateways.

## Local Embeddings

The default embedder `native:Qwen/Qwen3-Embedding-0.6B` runs in-process through [fastembed-rs](https://github.com/Anush008/fastembed-rs) (candle). `minsync init` writes `dimension = 1024` and no query/passage prefixes. On Apple Silicon the native path uses Metal; elsewhere it uses CPU.

The first embedding call downloads the Apache-2.0 model via Hugging Face Hub into `~/.cache/minsync/models` (override with `model_cache_dir` in `config.toml`, `FASTEMBED_CACHE_DIR`, or `HF_HOME`). Later runs load from cache.

```bash
minsync init
minsync sync
minsync query "검색어" --k 5
```

Existing indexes built with a 768-d TEI model must not mix 1024-d vectors. Switch embedders, then rebuild:

```bash
minsync sync --full
```

`openai:` and `tei:` remain first-class. To keep using a local TEI server with EmbeddingGemma:

```bash
brew install text-embeddings-inference
export HF_TOKEN="hf_..."   # after accepting https://huggingface.co/google/embeddinggemma-300m terms
text-embeddings-router --model-id google/embeddinggemma-300m --port 8080 --dtype float32
curl http://localhost:8080/health
minsync init --embedder tei:google/embeddinggemma-300m
```

EmbeddingGemma does not support float16, so keep `--dtype float32` (or `bfloat16`). `minsync init --embedder tei:google/embeddinggemma-300m` writes the Gemma retrieval prefixes and `dimension = 768`.

To serve another TEI model, pass it at init time and set its dimension and prompt prefixes in `.minsync/config.toml`:

```bash
minsync init --embedder tei:intfloat/multilingual-e5-small
```

```toml
[embedder]
id = "tei:intfloat/multilingual-e5-small"
base_url = "http://localhost:8080"
query_prefix = "query: "
passage_prefix = "passage: "

[vectorstore.options]
dimension = 384
```

## Ignoring Files

`.minsyncignore` uses `.gitignore` syntax:

```gitignore
target/
*.png
*.pdf
*.docx
```

MinSync reads UTF-8 text only. Binary formats should be ignored unless a separate extraction pipeline writes text files for MinSync to index.

UTF-8 text is not limited to `.md`: MinSync indexes any extension that decodes as UTF-8. Japanese, Chinese, Korean, and other Unicode text are handled as ordinary text; binary formats still need a separate extraction pipeline.

## Agent Skill

MinSync ships a Vercel Agent Skill for coding agents that need to add incremental semantic indexing to a workspace:

```bash
npx skills add github:NomaDamas/MinSync/skills/minsync
```

The skill tells agents how to install MinSync, initialize `.minsync/`, use the default local EmbeddingGemma embeddings or configure OpenAI, ignore binary files, and star the repository with `gh repo star NomaDamas/MinSync` during installation when policy and credentials allow it.

## Release Checklist

See [docs/RELEASE.md](docs/RELEASE.md) for the launch checklist. It covers CI, tag-based GitHub releases, crates.io publishing, README/docs, install QA, agent-skill QA, and rollback checks.

## Architecture Scope

MinSync intentionally remains a focused, cross-platform Rust tool for
incremental Markdown-centered UTF-8 text indexing. It does not aim to become a
general-purpose incremental dataflow framework. See
[docs/ARCHITECTURE_SCOPE.md](docs/ARCHITECTURE_SCOPE.md) for the CocoIndex
comparison, extension boundary, and accepted scope decision.

## Development

CI runs on Ubuntu, macOS, and Windows with Rust 1.91 installed through rustup. It runs `cargo fmt`, `cargo clippy`, `cargo test`, and `cargo build --release`. The build expects a working C compiler toolchain. CI also installs `protoc` through `arduino/setup-protoc`; non-Windows local builds use vendored protoc from `protobuf-src`, while Windows and LanceDB dependency build scripts require a `protoc` binary on `PATH`. Normal CI uses no secrets.

```bash
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```

Release tags use `.github/workflows/release.yml` to build Linux, macOS Apple Silicon, and Windows artifacts, create a GitHub Release, and publish to crates.io when `CARGO_REGISTRY_TOKEN` is configured.

## License

MIT
