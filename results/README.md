# Recorded results

Build names: `stock` is llama.cpp at the base commit; `all` is patches 0001 to 0004; `0001` is that patch alone;
`all+0005` adds the optional slot reservation. The section numbers refer to `docs/report.md`.

| Path | Base | Content | Report |
| --- | --- | --- | --- |
| `ab-holdout/stock-vs-all.json` | b94959a | held-out A/B, stock against 0001 to 0004 (`scripts/ab_compare.py` output) | 4.1 |
| `ab-holdout/stock-vs-0001.json` | b94959a | the same for 0001 alone, separate session | 4.1 |
| `ab-holdout/runs/` | b94959a | pooled raw runs behind the two files above: `stock-a` with `all`, `stock-b` with `0001`; one JSON line per request (query id, N, latency in ms, score per candidate, nDCG@10, round) | 4.1 |
| `kernel/` | b94959a | FA kernel A/B (`scripts/fa_kernel_ab.sh`, 6 rotated repetitions): `fa-kernel.log` has one line per run (build, repetition, K/V length, queries, sequences, prefix, us per run), `summary.txt` the table. Here `stock` is the base plus the test part of 0003 and `patched` is 0001 to 0004 | 4.6 |
| `openloop/` | b94959a | open loop, stock and `all`, one JSON line per arrival rate (`scripts/openloop.sh`) | 4.4 |
| `prefill/` | b94959a | llama-bench prefill, 8 rotated rounds: `stock-b94959a`, `wt-b94959a` (0004 alone) and `patch-b94959a` (0001 to 0004) | 4.7 |
| `mixed/` | b94959a | four mixed-load runs at 30 N=8 requests per second, `all` against `all+0005`; run 4 also has the same two builds on e4b620a | 4.5 |
| `e4b620a/*.json` | e4b620a | held-out A/B on the earlier base: `stock-vs-server` (0001), `stock-vs-cuda` (0003 and 0004), `server-vs-all`, `stock-vs-all`; prefixed `ub512-` with `-ub 512` and `defaults-` with the server defaults | 4.2, 4.3 |
| `e4b620a/mixed/` | e4b620a | mixed load, stock against `all` at 10 to 40 N=8 requests per second; `reserve-` files: `all` against `all+0005` | 4.5 |

In the A/B summaries, `a_*` is the first build and `b_*` the second; `p95_change` and `throughput_change` are
fractions, and `p95_change_ci` is the 95% bootstrap interval of the P95 change.
