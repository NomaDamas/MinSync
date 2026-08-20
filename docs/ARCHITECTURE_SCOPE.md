# Architecture Scope: Focused Incremental Text Indexing

- **Status:** Accepted
- **Date:** 2026-08-20

## Decision

MinSync will remain a small, efficient, cross-platform Rust library and CLI for
incrementally indexing Markdown-centered UTF-8 text.

Its core responsibility is to:

1. detect changed, added, and deleted files without git;
2. derive stable text chunks, with content-defined chunking (CDC) as the
   preferred strategy for preserving unchanged chunks across edits;
3. avoid recomputing unchanged chunk embeddings;
4. apply deterministic, crash-safe additions, updates, and deletions to an
   index; and
5. provide the same behavior on Linux, macOS, and Windows.

MinSync will not become a general-purpose incremental dataflow framework.

## Context

MinSync was compared with
[CocoIndex](https://github.com/cocoindex-io/cocoindex), which provides a broad
incremental execution framework for AI data pipelines. CocoIndex can track
arbitrary processing functions, dependencies, memoized results, and target
states. Parsing, chunking, BM25 preparation, sparse encoding, vector embedding,
image processing, and database writes can all be modeled as functions in such a
framework.

Expanding MinSync to support arbitrary parsers, modalities, transformation DAGs,
function memoization, and general target-state reconciliation would therefore
duplicate CocoIndex's scope while substantially increasing MinSync's runtime,
state-management, and extension complexity.

MinSync has a clearer advantage in a narrower domain: deterministic and
efficient incremental indexing of frequently edited text. Its manifest,
per-chunk identities, CDC boundaries, mark-and-sweep cleanup, and crash-safe
cursor advancement can remain understandable and inexpensive without requiring
a general dataflow runtime.

## Scope

### In scope

- Rust-native library and CLI APIs.
- Cross-platform behavior on Linux, macOS, and Windows.
- Git-free manifest-based file change detection.
- Markdown-centered UTF-8 text indexing.
- Compatibility with other files that decode as UTF-8.
- CDC, recursive, and Markdown-aware text chunking improvements.
- Stable chunk identities and reuse of unchanged chunk work.
- Embedding-provider adapters.
- Index or vector-store adapters that preserve MinSync's incremental contract.
- Deterministic deletion, stale-record sweeping, recovery, and verification.
- Performance work directly related to text scanning, chunking, embedding, and
  index mutation.

### Out of scope

- A general-purpose transformation DAG or workflow engine.
- Arbitrary function memoization and dependency tracking.
- General ETL features such as joins, aggregation, and window processing.
- A universal parser or extraction framework for PDF, DOCX, images, audio, or
  video.
- Visual document retrieval pipelines and page-image lifecycle management.
- GPU worker scheduling or distributed pipeline orchestration.
- A generic target-state framework for non-indexing workloads.
- Reimplementing CocoIndex under a MinSync-specific API.

Binary-document support should remain external: another tool may extract or
render content and write UTF-8 text for MinSync to index.

## Extension Boundary

MinSync may expose Rust traits and adapter crates for embedders, chunkers, and
index stores. These extensions must fit the existing text-indexing lifecycle:

```text
scan text files
  -> detect changes
  -> normalize and chunk text
  -> reuse unchanged chunks
  -> compute missing embeddings
  -> apply index mutations
  -> advance crash-safe state
```

An extension is appropriate when it changes how one of these steps is
implemented without turning MinSync into a framework for arbitrary processing
graphs.

For example, a new vector database adapter is in scope. A system that lets users
compose PDF rendering, OCR, image embedding, joins, and arbitrary targets into a
memoized DAG is not.

## Why CDC Remains Important

CDC chooses chunk boundaries from content rather than fixed offsets. Small
insertions near the start of a Markdown file can therefore leave later chunk
boundaries and content hashes unchanged. MinSync can reuse those chunks instead
of embedding the entire file again.

CDC is not required for every text workload, so recursive and future
Markdown-aware chunkers may remain available. It is nevertheless a key
differentiator for MinSync's chosen workload: frequently edited text where
incremental embedding cost matters.

## Consequences

### Benefits

- A smaller API and dependency surface.
- Predictable cross-platform operation.
- Easier correctness and crash-recovery testing.
- Clear optimization goals and benchmarks.
- Less overlap with CocoIndex.
- A strong reason to use MinSync when only efficient incremental text indexing
  is required.

### Trade-offs

- MinSync will not natively cover every indexing modality.
- Users needing arbitrary AI data pipelines should use CocoIndex or another
  general dataflow framework.
- Binary documents require an external extraction step.
- Some possible integrations will be rejected when they expand the product
  beyond incremental text indexing.

## Reconsideration

This decision should be reconsidered only if the focused text-indexing model
cannot support MinSync's users, not merely because a broader pipeline is
technically possible. Any proposal to broaden the scope must demonstrate a
specific user need that cannot be met through a text-producing preprocessing
step or a bounded adapter.
