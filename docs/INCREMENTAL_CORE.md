# Incremental core: durable item state and crash-resume

`src/incremental/` holds the part of MinSync that decides *what still has to
be done* and *what is already done*, independent of what a stage produces. Two
stages use it today:

- the **vector stage** (`minsync sync`): chunk, embed, write to the store;
- a **materialization stage** (`minsync stage`): an external host (a parser)
  turns source documents into deterministic text files.

The core is not a workflow engine. It manages stable items and durable stage
completion, nothing else. See [ARCHITECTURE_SCOPE.md](ARCHITECTURE_SCOPE.md).

## Layout

| Module | Responsibility |
|---|---|
| `journal` | run header, append-only per-item records, the loader that decides whether a ledger can be trusted |
| `plan` | pure resume planner: one decision per item (`process`, `retry`, `skip`, `delete`, `preserve`) |
| `snapshot` | committed state of a materialization stage; written once per run |
| `item` | item-key and output-path validation at the host boundary |
| `stage` | `StageRun`: lock, run marker, journal, commit validation, compaction |
| `protocol` | the JSONL session `minsync stage` speaks on stdin/stdout |

The core imports only `crate::error` and `crate::state` (lock and run marker).
It does not import embedders, chunkers, vector stores, query code or parsers.
The vector stage keeps its own manifest and cursor (the native filesystem
adapter) and shares the journal and the resume status check; the
materialization stage uses `plan`, `snapshot` and `stage` on top of it.

## The commit order

```text
read source -> compute output -> durably commit output -> append completion record
```

A completion record is appended only after the output commit returned.

- **Vector stage.** `sync` writes a batch to the store (`index_batch`), and only
  then appends the batch's records (`completion_records`) with one fsync.
- **Materialization stage.** The host writes the output next to its final path
  and renames it into place. `StageRun::commit` then fsyncs the file, checks
  it, and only then appends the record. A refused commit appends nothing.

The guarantee is at-least-once: a crash can repeat work, never turn a missing
output into a completed item. Output commits therefore have to be idempotent
(chunk ids are deterministic; a host re-renaming the same output is harmless).

## Ledger (`progress.jsonl`)

Line 1 is a header, every further line one record.

```json
{"format":2,"source_id":"…","stage":"vector","run_token":"…","full":false,"baseline":"sha256:…","fingerprint":"sha256:…","created_at":"…"}
{"item":"notes/a.md","version":"sha256:…","outcome":"committed"}
{"item":"notes/gone.md","outcome":"deleted"}
{"item":"docs/b.pdf","version":"sha256:…","outcome":"failed","error":"parser crashed"}
{"item":"docs/c.pdf","version":"sha256:…","outcome":"committed","output":{"path":"docs/c.pdf.md","size":812,"digest":"sha256:…","mtime_ns":1761300000000000000}}
```

A ledger is resumed only when `source_id`, `stage`, `full`, `baseline` (hash of
the committed manifest/snapshot the run diffed against) and `fingerprint` (hash
of every setting that changes the output) all match the new run. The token and
creation time are carried over; they identify the run, not its meaning.

Loading is deliberately suspicious:

| Situation | Result |
|---|---|
| no file | redo |
| torn final line (no newline) | ignored; earlier records are kept |
| malformed line anywhere else, empty file, header missing or malformed, unreadable file | whole ledger discarded, redo |
| `format` newer than this build | discarded, redo |
| `committed` record without a version | discarded, redo |

A `failed` record never completes an item and does not erase an earlier
`committed` record of the same item (that output still exists and has to be
reconciled); a later `committed` or `deleted` record clears the failure. When a
run starts, the ledger is rewritten atomically (header plus carried records:
completions, then failures, each in item order), which also drops a torn tail
instead of appending to it.

Ledgers written before the core existed (`path` / `content_hash` records and a
`sync_token` / `manifest_hash_from` / `config_fingerprint` header) are read
through serde aliases, so an interrupted run resumes after an upgrade. A
ledger written by this build is not readable by an older one; the older one
treats it as absent and redoes the work.

## Resume decisions

For each current item the planner compares it with the committed state and the
interrupted run's ledger (the ledger wins):

| Decision | When |
|---|---|
| `skip` | committed for this source version **and** this fingerprint, and the output still verifies. In a `full` run only completions of this run's own ledger count. |
| `process` | new, changed since its completion, fingerprint changed, or the recorded output is missing or changed |
| `retry` | failed before for exactly this source version |
| `delete` | has a completion, is absent from the source, and is not listed as unobservable |
| `preserve` | listed as unobservable: previous output kept, never deleted |

An output verifies when it is a regular file of the recorded size and either
its mtime equals the recorded one or its digest does. `"verify": "digest"`
always re-hashes, which also catches a change that kept size and mtime.

## Materialization stage

State lives in one directory per stage and corpus (`--state-dir`):

| File | Purpose |
|---|---|
| `lock` | held for the whole session; the OS releases it on any exit, including `kill -9` |
| `txn.json` | run marker |
| `progress.jsonl` | the journal of the running or interrupted run |
| `snapshot.json` | committed state: per item the version, fingerprint and output identity, plus the latest unresolved failures |

`finish` folds the ledger into the snapshot with one atomic write and then
removes the journal and the marker. The snapshot is therefore never rewritten
per item: a checkpoint costs one appended record, independent of corpus size.
A snapshot that exists but cannot be parsed is an error, not an empty baseline
(an empty baseline would orphan every output); a state directory of another
corpus or stage is refused (`source_mismatch`).

### Session protocol

`minsync stage --state-dir DIR` reads one JSON request per line on stdin and
writes one JSON response per line on stdout. One session is one run.

```text
-> {"op":"open","stage":"parse","source_id":"corpus","fingerprint":"parser-1",
    "output_root":"/data/mirror",
    "items":[{"key":"a.pdf","version":"sha256:…"}],
    "unknown":["b.pdf"],"full":false,"verify":"fast","wait":false}
<- {"ok":true,"op":"open","run_token":"…","resumed":false,
    "counts":{"process":1,"retry":0,"skip":0,"resumed":0,"delete":0,"preserve":1},
    "decisions":[{"key":"a.pdf","action":"process","version":"sha256:…"}]}

-> {"op":"commit","key":"a.pdf","version":"sha256:…",
    "output":{"path":"a.pdf.md","size":123,"digest":"sha256:…"}}
<- {"ok":true,"op":"commit","key":"a.pdf","output":{"path":"a.pdf.md","size":123,"digest":"sha256:…"}}

-> {"op":"fail","key":"c.pdf","version":"sha256:…","error":"parser crashed"}
-> {"op":"delete","key":"gone.pdf"}
<- {"ok":true,"op":"delete","key":"gone.pdf","output_removed":true}
-> {"op":"finish"}
<- {"ok":true,"op":"finish","committed":1,"deleted":1,"failed":1,"pending":1}
```

- `items` is the whole current source, key and version each. An item absent
  from `items` and from `unknown` is treated as confirmed gone. Keys are
  normalized relative identifiers (no empty, `.` or `..` component, no
  backslash, no control characters, at most 4096 bytes) or host document ids.
- `decisions` lists only work to do (`process`, `retry`, `delete`); skips and
  preserved items are counted in `counts`.
- `commit` takes `fingerprint` optionally; it must equal the run's. `size` and
  `digest` are optional cross-checks, MinSync always measures the file and
  returns what it recorded.
- `delete` removes the recorded output (idempotent) and records the deletion.
  It refuses an item that is still in `items` (`still_present`) or listed in
  `unknown` (`preserved_item`).
- `finish` leaves failed and untouched items with their previous completion;
  `pending` counts what the next run will still do.

A refused request is answered `{"ok":false,"op":"…","error":{"code":"…","message":"…"}}`
and the session continues. Codes: `invalid_request`, `invalid_item_key`,
`duplicate_item`, `not_open`, `already_open`, `unknown_item`,
`version_mismatch`, `fingerprint_mismatch`, `invalid_output_path`,
`output_missing`, `output_not_file`, `output_mismatch`, `output_conflict`,
`still_present`, `preserved_item`, `source_mismatch`, `corrupt_state`,
`lock_failed`, `io_error`, `internal`.

`commit` accepts an output only if the path is relative with plain components
only, the file is a regular file (not a symlink) whose real location is inside
`output_root`, no other item owns that path, and the claimed size and digest
match. The file is opened once without following a symlink, and its size, mtime
and digest are taken from that one descriptor. The output root is host-managed:
MinSync guards against a buggy host (traversal, stray symlinks, collisions) but
not against another process that rewrites directories under `output_root`
while a run is in progress (a parent directory swapped for a symlink between
validation and `open`/`delete`). Output paths of one item should stay the same
across runs; an item that moves to a new path leaves its old file behind.

Exit codes: `0` after `finish`, `3` when another run holds the lock (a
`lock_failed` response first), `1` when stdin ends before `finish`. In that
case the run stays resumable: the next `open` reports `resumed: true` and plans
only the items without a completion.

## Failure semantics

| Event | Result |
|---|---|
| crash while processing an item | no record; the item is pending next run; earlier completions stay valid |
| crash after the output commit, before the record | the item is processed again; the idempotent commit repairs the missing record |
| crash after the record | skipped only while the source version, fingerprint and output verification still match |
| crash during compaction | either the old snapshot plus a resumable journal, or the new snapshot (a stale journal then fails the `baseline` check and is discarded) |
| source unreadable | listed in `unknown`: previous output kept, never deleted; retried later |
| source gone and confirmed | `delete`: output removed, deletion recorded |
| processing fingerprint changed | old completions are not reused; every present item is processed under the new fingerprint, a failed one keeps its old completion and output |
| processing failed for one item | recorded as failed; its previous output is intact; other items are unaffected |
| host dies / stdin closes | exit 1, run resumable |
| MinSync killed (`SIGKILL`, power loss) | the OS drops the lock; torn final record ignored; resumable |
| journal corrupt or from another baseline/fingerprint | discarded; work redone, nothing assumed complete |

Cancellation does not need a separate path: state is consistent at every
instruction boundary, so any signal or kill leaves a resumable journal and a
released lock. Graceful interruption of `minsync sync` (#85) can build on this
unchanged.

## Not included

- Parser implementations. The host parses; MinSync records.
- Wiring a specific host (AutoRAG parsing) to the protocol.
- Bounding or compacting the journal *during* a run, progress events for the
  stage protocol, and streaming iteration of very large source listings. The
  journal grows by one line per completed item and is compacted when the run
  finishes.
