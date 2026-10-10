# `bench_issue71` — MinSync issue #71 benchmark harness

Reproducible harness comparing a **baseline** MinSync binary against a
**candidate** binary on the issue's real condition: a LanceDB store left with
thousands of fragments/versions by sync runs that were killed before the final
`flush()`. On such a store every chunk is its own commit and every filter scan
touches all fragments, so each file costs seconds.

Nothing touches `src/`. Standard library only (`python3`), no pip.

## Files

| File | Purpose |
|---|---|
| `bench.py` | Entrypoint: builds workspaces, runs every scenario, prints a markdown report |
| `fake_tei.py` | Deterministic fake TEI embedding server (`POST /embed`, `GET /stats`, `POST /reset`, `GET /health`) |
| `README.md` | This file |

`fake_tei.py` returns sha256-derived, L2-normalised, 16-dimensional vectors, so
both binaries build byte-identical indexes for identical workloads with no real
model. `bench.py` starts it on a free port and rewrites each workspace's
`.minsync/config.toml` (`[embedder].base_url`, `batch_size`, `max_concurrent`,
`[vectorstore.options].dimension = 16`).

## Quick start

```bash
# baseline vs baseline (no candidate yet): the safe smoke test
python3 scripts/bench_issue71/bench.py \
    --baseline /tmp/bench/minsync-main \
    --candidate /tmp/bench/minsync-main \
    --scenario all \
    --n 120 --repeats 1 \
    --timeout 180 --kill-timeout 0.5 --fragments 300 --frag-budget 180 \
    --out /tmp/results.md

# real comparison once the candidate exists
python3 scripts/bench_issue71/bench.py \
    --baseline /tmp/bench/minsync-main \
    --candidate target/release/minsync \
    --scenario all \
    --out scripts/bench_issue71/results.md
```

Defaults: `--n 600`, `--repeats 3` (scenarios 1–2), `--timeout 900`,
`--kill-timeout 3`, `--fragments 3000`, `--latency-ms 0`, `--dim 16`,
`--batch-size 64`, `--max-concurrent 1`.

## Scenarios (exact commands)

Run one at a time with `--scenario <name>`, or all with `--scenario all`
(the default).

1. **initial** — fresh initial sync of `--n` files (default 600).

   ```bash
   python3 scripts/bench_issue71/bench.py --scenario initial --candidate /tmp/bench/minsync-main
   ```

2. **incremental** — one baseline initial sync, then `--incremental-frac`
   (default 0.30) of files get one same-length paragraph edit; time the
   incremental sync on both binaries.

   ```bash
   python3 scripts/bench_issue71/bench.py --scenario incremental --candidate target/release/minsync
   ```

3. **fragmented** — the issue's condition. Repeatedly mutates the corpus and
   runs the **baseline** under a SIGKILL (`--kill-timeout`), until the store has
   `--fragments` (default 3000) fragments/versions, then snapshots the workspace
   (`cp -R`, `.minsync` included) and runs both binaries to completion on
   separate copies with `--timeout`.

   ```bash
   python3 scripts/bench_issue71/bench.py --scenario fragmented \
       --candidate target/release/minsync \
       --fragments 3000 --kill-timeout 3 --timeout 900
   ```

4. **resume** — candidate only. Runs the candidate uninterrupted (records
   embedded-text count and final row count), then runs it again, SIGKILLs it
   once `--resume-kill-frac` (default 0.4) of the embeddings are done, reruns to
   completion, and asserts total embedded work `< 1.5x` the uninterrupted run
   and that the final row count matches. Reports `PASS`/`FAIL` clearly (the
   candidate may not implement a resume feature yet).

   ```bash
   python3 scripts/bench_issue71/bench.py --scenario resume --candidate target/release/minsync
   ```

## Metrics

Per run: wall seconds of the `minsync sync` process, exit code, the parsed
`--format json` sync result (`chunks_added/updated/deleted`, `files_*`,
`embedded_texts`, `elapsed_seconds`), and store health from the filesystem under
`.minsync/store/documents.lance` (file counts in `_versions/`, `data/`,
`_deletions/` and total size). After each run, `minsync query --format json` is
run in `bm25` and `vector` modes as a sanity check, and the set of
`(path, doc_id)` pairs is compared between the two binaries.

Scenarios 1–2 repeat `--repeats` times and report the **median** wall time plus
a speedup ratio (baseline median / candidate median). The report is printed to
stdout and written to `--out`.

## Caveats

- **Probe cursor.** A store left by a killed run has no `cursor.json`, so
  `minsync query` refuses to read it. For post-hoc row counting on such a
  workspace the harness writes a throwaway `cursor.json` (correct `source_id`,
  placeholder hash), queries, then deletes it again. It is never present during
  a timed run, so it cannot affect sync behaviour — only the measurement.
- **ANN disabled on purpose.** Workspaces set `[vectorstore.options]`
  `index_build_threshold`/`index_optimize_delta_threshold` very high
  (`--ann-build-threshold`, default 1e9) so the vector search stays exhaustive.
  With the default 256-row threshold LanceDB builds an approximate IVF-HNSW
  index, and a large-k query then returns a *different row count on each run*
  (~660 vs ~662), which would make the `(path, doc_id)` comparison flaky.
  Disabling ANN also removes run-to-run index-maintenance cost from the sync
  timing. Exact row counts therefore come from the exhaustive vector query; pass
  a smaller `--ann-build-threshold` to benchmark with ANN enabled.
- **`bm25` is a sanity check, not a row count.** LanceDB full-text search for a
  common token can return fewer rows than the store holds (observed 462/662 on
  an incrementally-updated store, identical for both binaries); the harness
  only requires both binaries to agree and to not error.
- Wall times include process spawn and LanceDB open; the internal
  `elapsed_seconds` field is reported too.
- `resume` measures **embedded work**, not wall time: the second (resumed) run
  can be much slower than the first because it faces the fragmented store from
  the killed run — that is issue #71, not a resume failure.
- The fragmented scenario's baseline may legitimately time out. The report then
  shows `TIMEOUT` and the fragments/versions it left behind (partial progress
  rows come from the probe-cursor row count / fragment counts).
- `cp -R` of a fragmented store is not timed; only the `minsync sync` process is.
- Results are single-machine and I/O-bound; use `--latency-ms` to add artificial
  embedding latency when comparing CPU-bound paths.
