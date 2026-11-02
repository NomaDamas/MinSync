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
| `snapshot` | committed state of a materialization stage: completions, omissions and failures; written once per run |
| `item` | item-key and output-path validation at the host boundary |
| `stage` | `StageRun`: lock, run marker, journal, commit validation, compaction |
| `view` | read-only state reconstruction: `plan` and `inspect`, with no lock, no mutation |
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
{"format":3,"source_id":"…","stage":"vector","run_token":"…","full":false,"baseline":"sha256:…","fingerprint":"sha256:…","created_at":"…"}
{"item":"notes/a.md","version":"sha256:…","outcome":"committed"}
{"item":"notes/gone.md","outcome":"deleted"}
{"item":"docs/b.pdf","version":"sha256:…","outcome":"failed","error":"parser crashed"}
{"item":"docs/c.pdf","version":"sha256:…","outcome":"committed","output":{"path":"docs/c.pdf.md","size":812,"digest":"sha256:…","mtime_ns":1761300000000000000}}
{"item":"docs/d.pdf","version":"sha256:…","outcome":"skipped","reason":"unsupported type"}
{"item":"docs/e.pdf","version":"sha256:…","outcome":"failed","error":"bad xref","deferred":true,"retry_fingerprint":"retry-1"}
{"item":"docs/f.pdf","version":"sha256:…","outcome":"failed","error":"corrupt","discard_output":true}
```

| Field | Meaning |
|---|---|
| `version` | the source version the work, omission or failure is about; required for `committed` and `skipped`, absent for `deleted` |
| `output` | the committed file's identity: path relative to the output root, size, `sha256:` digest, and the mtime at commit in nanoseconds (a JSON integer) |
| `reason` | why a `skipped` item is deliberately omitted |
| `deferred` | a `failed` item is held: it is not retried while its source version, the run fingerprint and the retry fingerprint all stay the same |
| `retry_fingerprint` | the `retry_fingerprint` of the run that deferred the failure |
| `discard_output` | the failure threw the item's previous completion and output away instead of leaving the last good output in place |

A ledger is resumed only when `source_id`, `stage`, `full`, `baseline` (hash of
the committed manifest/snapshot the run diffed against) and `fingerprint` (hash
of every setting that changes the output) all match the new run. The token and
creation time are carried over; they identify the run, not its meaning.
`retry_fingerprint` is not part of the run's identity: it decides whether a
held failure is released, not whether this ledger belongs to this run.

Loading is deliberately suspicious:

| Situation | Result |
|---|---|
| no file | redo |
| torn final line (no newline) | ignored; earlier records are kept |
| malformed line anywhere else, empty file, header missing or malformed, unreadable file | whole ledger discarded, redo |
| `format` newer than this build | discarded, redo |
| `committed` or `skipped` record without a version | discarded, redo |

A `failed` record never completes an item and does not erase an earlier
`committed` record of the same item (that output still exists and has to be
reconciled) unless it is a `discard_output` failure, which does erase it; a
later `committed`, `deleted` or `skipped` record clears the failure, and a
`committed` or `deleted` record clears an earlier omission. When a run starts,
the ledger is rewritten atomically (header plus carried records: completions,
then omissions, then failures, each in item order), which also drops a torn tail
instead of appending to it.

Ledgers written before the core existed (`path` / `content_hash` records and a
`sync_token` / `manifest_hash_from` / `config_fingerprint` header) are read
through serde aliases, so an interrupted run resumes after an upgrade; a
format-2 ledger is read the same way. A ledger written by this build is not
readable by an older one; the older one treats it as absent and redoes the
work.

## Resume decisions

For each current item the planner compares it with the committed state and the
interrupted run's ledger (the ledger wins), in this order:

| Decision | When |
|---|---|
| `skip` (completion) | committed for this source version **and** this fingerprint, and the output still verifies. In a `full` run only completions of this run's own ledger count. |
| `skip` (omission) | deliberately omitted for this source version and fingerprint. No output is needed, and a `full` run does not invalidate it: the host bakes its omission policy into the source version, so a policy change changes the version. |
| `skip` (held failure) | a deferred failure for exactly this source version, run fingerprint and retry fingerprint, and the run is neither `full` nor `retry_failed`. The item keeps its last good output; the decision carries `"deferred": true` and the failure's `error`. |
| `process` | new, changed since its completion, fingerprint changed, or the recorded output is missing or changed |
| `retry` | failed before for exactly this source version (a released deferred failure included) |
| `delete` | has a completion, an omission or a failure, is absent from the source, and is not listed as unobservable |
| `preserve` | listed as unobservable: previous state kept, never deleted |

`full` releases every deferred failure and reuses only what the run itself
committed; `retry_failed` releases deferred failures while still reusing the
committed state.

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
| `snapshot.json` | committed state: per item the version, fingerprint and output identity, the deliberate omissions with their reason and fingerprint, and the latest unresolved failures with their fingerprint and deferral |

`finish` folds the ledger into the snapshot with one atomic write and then
removes the journal and the marker. The snapshot is therefore never rewritten
per item: a checkpoint costs one appended record, independent of corpus size.
A snapshot that exists but cannot be parsed is an error, not an empty baseline
(an empty baseline would orphan every output); a state directory of another
corpus or stage is refused (`source_mismatch`). A failure or an omission is
dropped at `finish` when its item is no longer in the source.

Snapshot format 1 (only completions and failures) still loads. Format 2 adds
`skipped` and the deferred-failure fields; an older MinSync refuses it with
`corrupt_state` instead of misreading the stage's durable state. `finish`
always saves the current format, so a legacy snapshot that picks up an omission
or a deferred failure is upgraded rather than left under a format that claims
it cannot hold them.

### Session protocol

`minsync stage --state-dir DIR` reads one JSON request per line on stdin and
writes one JSON response per line on stdout. One session is one run.

```text
-> {"op":"open","stage":"parse","source_id":"corpus","fingerprint":"parser-1",
    "output_root":"/data/mirror",
    "items":[{"key":"a.pdf","version":"sha256:…"}],
    "unknown":["b.pdf"],"full":false,"verify":"fast","wait":false,
    "retry_fingerprint":"retry-1","retry_failed":false}
<- {"ok":true,"op":"open","run_token":"…","resumed":false,
    "counts":{"process":1,"retry":0,"skip":0,"resumed":0,"delete":0,"preserve":1},
    "decisions":[{"key":"a.pdf","action":"process","version":"sha256:…"}]}

-> {"op":"commit","key":"a.pdf","version":"sha256:…",
    "output":{"path":"a.pdf.md","size":123,"digest":"sha256:…"}}
<- {"ok":true,"op":"commit","key":"a.pdf","output":{"path":"a.pdf.md","size":123,"digest":"sha256:…"}}

-> {"op":"fail","key":"c.pdf","version":"sha256:…","error":"parser crashed",
    "defer":false,"discard_output":false}
-> {"op":"skip","key":"d.pdf","version":"sha256:…","reason":"unsupported type"}
<- {"ok":true,"op":"skip","key":"d.pdf"}
-> {"op":"delete","key":"gone.pdf"}
<- {"ok":true,"op":"delete","key":"gone.pdf","output_removed":true}
-> {"op":"finish"}
<- {"ok":true,"op":"finish","committed":1,"deleted":1,"failed":1,"pending":1}
```

- `items` is the whole current source, key and version each — including items
  the host's own policy omits, which it then reports with `skip`. An item
  absent from `items` and from `unknown` is treated as confirmed gone. Keys are
  normalized relative identifiers (no empty, `.` or `..` component, no
  backslash, no control characters, at most 4096 bytes) or host document ids.
- `version` is opaque to MinSync: the host's source-mapping JSON (virtual and
  source path, parser name and version, source mtime and size, source version),
  or any string that changes whenever the meaning changes. A host that encodes
  its omission policy in the version makes a policy change invalidate every
  omission by itself.
- `decisions` lists work to do (`process`, `retry`, `delete`) plus, so the host
  can tell them from items that are simply up to date, the skips that rest on a
  deferred failure, as `{"key":…,"action":"skip","version":…,"deferred":true,
  "error":"…"}`. Other skips and preserved items are counted in `counts`, and a
  held failure is counted in `skip`.
- `retry_fingerprint` identifies the host's retry policy; `retry_failed` asks
  for held failures to be retried after all. Either one changing releases them.
- `commit` takes `fingerprint` optionally; it must equal the run's. `size` and
  `digest` are optional cross-checks, MinSync always measures the file and
  returns what it recorded.
- `fail` records a failure and keeps the item's previous output, unless
  `discard_output` is set, in which case MinSync removes that output and the
  completion that pointed at it. `defer` holds the failure.
- `skip` records a deliberate omission. It requires the item to be part of the
  run (`unknown_item` otherwise) and removes any output a previous completion
  recorded.
- `delete` removes the recorded output (idempotent) and records the deletion.
  It refuses an item that is still in `items` (`still_present`) or listed in
  `unknown` (`preserved_item`); an item that only holds a failure or an
  omission is deleted to clear it. An item that holds nothing is answered
  `output_removed: false` and records nothing, so it is not counted in
  `finish`'s `deleted`.
- `finish` leaves failed and untouched items with their previous completion;
  `pending` counts what the next run will still do, so a held failure is not
  pending.

`plan` and `inspect` are read-only, need no `open`, and are also accepted in
the middle of a session. Neither takes the stage lock, creates the state
directory, nor writes a marker, journal or snapshot.

`capabilities` is a read-only request of the same kind, and the marker a host
must check first:

```text
-> {"op":"capabilities"}
<- {"ok":true,"op":"capabilities","protocol":2,
    "features":["retry_fingerprint","retry_failed","deferred_failure",
                "discard_output","deliberate_skip","plan","inspect"],
    "journal_format":3,"snapshot_format":2}
```

A host must ask for it — and refuse to run when the request is refused — before
sending `retry_fingerprint`, `retry_failed`, `defer`, `discard_output`, `skip`,
`plan` or `inspect`. An older MinSync parses `open` and `fail` with unknown
*fields* ignored, so it would silently hold nothing, discard nothing and retry
at the wrong time. The newer *ops* and the newer on-disk formats do not need the
probe to fail loudly: an unknown op is refused with `invalid_request`, a
format-2 snapshot is refused with `corrupt_state`, and a format-3 ledger is
discarded and redone.

```text
-> {"op":"plan","stage":"parse","source_id":"corpus","fingerprint":"parser-1",
    "output_root":"/data/mirror","items":[…]}
<- {"ok":true,"op":"plan","counts":{…},
    "decisions":[{"key":"a.pdf","action":"skip","version":"sha256:…"},
                 {"key":"c.pdf","action":"skip","version":"sha256:…",
                  "deferred":true,"error":"parser crashed"}]}

-> {"op":"inspect","stage":"parse","source_id":"corpus",
    "output_root":"/data/mirror","verify":"fast"}
<- {"ok":true,"op":"inspect",
    "items":[{"key":"a.pdf","version":"sha256:…","fingerprint":"parser-1",
              "output":{"path":"a.pdf.md","size":123,"digest":"sha256:…",
                        "mtime_ns":"1761300000000000000"},
              "updated_at":"2026-10-24T09:20:00+00:00"}],
    "failed":[{"key":"c.pdf","version":"sha256:…","error":"parser crashed",
               "deferred":true},
              {"key":"e.pdf","version":null,"error":"bad xref",
               "deferred":false}],
    "skipped":[{"key":"d.pdf","version":"sha256:…","reason":"unsupported type",
                "fingerprint":"parser-1"}],
    "known":[{"key":"a.pdf","version":"sha256:…"},
             {"key":"c.pdf","version":"sha256:…"},
             {"key":"d.pdf","version":"sha256:…"},
             {"key":"e.pdf","version":null}]}
```

- `plan` takes the `open` field set and returns the same planner's decisions
  for every item, so the host can report staleness; `open` returns only the
  work plus the held-failure skips.
- `inspect` reconstructs the durable state: committed items (with the output
  identity that still verifies), every held or resolved failure, and every
  omission, each with its source version for the host's own metadata mapping.
  It merges the snapshot with the journal of the run that was interrupted or is
  still active, and trusts that journal only when it describes the same stage
  and corpus and diffed against the snapshot's own baseline — never an
  unrelated run's. Because a `finish` rewrites the snapshot and then removes
  the journal, the snapshot is read on both sides of the journal and a read
  that straddled a finish is retried, so a host is never handed the state that
  the run it just missed replaced; a read that keeps straddling finishes
  answers from the snapshot alone, which is a state the stage really held and
  whose outputs are verified anyway. `items[].output.mtime_ns` is a decimal
  string, because a nanosecond timestamp does not survive a JSON number in a
  JavaScript host; on disk it stays a JSON integer. `updated_at` is a stable,
  per-output RFC 3339 timestamp derived from that recorded mtime.
- `inspect` also returns `known`, always present: the roster of every key the
  durable snapshot and journal hold — every completion, omission and failure —
  whether or not its output still verifies and whether or not its failure is
  deferred. A host uses it to preserve the state of roots it cannot currently
  read, never as a substitute for a verified output; the `items`, `failed` and
  `skipped` lists keep their own semantics. A key's `version` comes from its
  failure if it has one, else its omission, else its completion; `null` marks a
  failure recorded before versions were kept.

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
Everything that removes an output (`delete`, `skip`, a `discard_output`
failure) goes through the same checks, refuses a symlinked parent that would
leave the root, and tolerates a file that is already gone.

Exit codes: `0` after `finish`, `3` when another run holds the lock (a
`lock_failed` response first, whose `error.holder` gives the holder's `pid`,
`started_at` and `command` when the lock file names them), `1` when stdin ends before `finish`. In that
case the run stays resumable: the next `open` reports `resumed: true` and plans
only the items without a completion. A session that only sends
`capabilities`, `plan` and/or `inspect` holds nothing and exits `0` when stdin
closes, which is how a host can check the binary and read the durable state
without a run at all.

## Failure semantics

| Event | Result |
|---|---|
| crash while processing an item | no record; the item is pending next run; earlier completions stay valid |
| crash after the output commit, before the record | the item is processed again; the idempotent commit repairs the missing record |
| crash after the record | skipped only while the source version, fingerprint and output verification still match |
| crash during compaction | either the old snapshot plus a resumable journal, or the new snapshot (a stale journal then fails the `baseline` check and is discarded) |
| source unreadable | listed in `unknown`: previous state kept, never deleted; retried later |
| source gone and confirmed | `delete`: output removed, and any completion, omission or failure cleared |
| processing fingerprint changed | old completions are not reused; every present item is processed under the new fingerprint, a failed one keeps its old completion and output, an omitted one is processed again, and a held failure is released |
| retry fingerprint changed, `retry_failed`, or `full` | held failures are released and retried |
| processing failed for one item | recorded as failed; its previous output is intact; other items are unaffected |
| a failure deferred by the host | held: not retried, not pending, reported as a skip decision with its error; the last good output stays |
| a failure with `discard_output` | the previous output is removed and its completion dropped, so nothing points at a file that is gone |
| an item deliberately omitted (`skip`) | recorded separately from a failure, any previous output removed, and reused with no output while its source version and fingerprint hold |
| host dies / stdin closes | exit 1, run resumable |
| MinSync killed (`SIGKILL`, power loss) | the OS drops the lock; torn final record ignored; resumable |
| journal corrupt or from another baseline/fingerprint | discarded; work redone, nothing assumed complete |
| `inspect` or `plan` while a run finishes, or while another holds the lock | reads no lock; reconstructs from the snapshot plus the run's own journal and reports the durable truth, never a half-written one |

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
