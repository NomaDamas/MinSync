# Issue #71 benchmark results (main 9ff586f vs this branch)

Fake TEI embedder (dim 16, 0 ms latency), N=600 files (~8 chunks each), macOS arm64, release builds. Median of 3 for initial/incremental.

| Scenario | main (s) | PR (s) | Speedup | versions after (main -> PR) |
|---|---:|---:|---:|---|
| initial sync | 32.50 | 13.04 | 2.49x | 1806 -> 630 |
| incremental (30% files edited) | 65.88 | 10.02 | 6.57x | 3607 -> 2357 |
| fragmented store (2450 frag / 3089 ver left by killed syncs) | 137.25 | 31.52 | 4.35x | 5778 -> 3983 |
| resume after SIGKILL (PR only) | n/a | embedded work 1.000x of uninterrupted run, same row count | | |

Row sets (path, chunk id) were identical between binaries in every scenario (4798 rows).
Reproduce: python3 scripts/bench_issue71/bench.py --baseline <main-binary> --candidate <pr-binary> --scenario all
