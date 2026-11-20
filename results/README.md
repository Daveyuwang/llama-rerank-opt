# Recorded results

All measurements ran on llama.cpp b94959a with the patches in this repository, on an RTX 5080 under WSL2. Server
names: `stock` is the base; `all` is patches 0001 to 0004; `server` is 0001 and 0002; `cuda` is 0003 and 0004; `wt`
is 0004 alone; `vllm` is vLLM 0.34.0 with its default prefix caching and `vllm-noapc` without it. For the kernel
A/B, `stock` carries the test part of 0003. The section numbers refer to `docs/report.md`.

| Path | Content | Report |
| --- | --- | --- |
| `ab-holdout/runs/` | pooled held-out runs of stock, all, server, cuda and vllm: one JSON line per request (query id, N, latency in ms, score per candidate, nDCG@10, round, workload hash) | 4.1 to 4.3 |
| `ab-holdout/ab-*.json` | `scripts/ab_compare.py` summaries: each server against stock, and all against vllm | 4.1 to 4.3 |
| `ab-holdout/log.txt` | console output of the A/B, with the prompt tokens processed per session | 4.1 |
| `ab-holdout/prompt-tokens-n32.txt` | prompt tokens processed in one round of N=32 requests | 4.1 |
| `configs/` | A/B summaries of other configurations: `-ub 512`, the server defaults, split KV with and without prefix sharing, vLLM without prefix caching | 4.3, 4.5 |
| `model-4b/` | A/B summaries with Qwen3-Reranker-4B | 4.4 |
| `openloop/` | open loop of stock and all, one JSON line per arrival rate | 4.6 |
| `mixed/` | mixed load of stock and all, one file per server, N=8 rate and request size; `log.txt` has the table | 4.7 |
| `kernel/` | FA kernel A/B: `fa-kernel.log` has one line per run (build, repetition, K/V length, queries, sequences, prefix, us per run), `summary.txt` the table | 4.8 |
| `prefill/` | llama-bench prefill of stock, wt and all with `-ub 2048` and `-ub 512`: one JSON line per build, round and prompt length; the `.txt` files have the tables | 4.9 |
| `tests/` | summary lines of the FA tests and the server tests | 6 |

In the A/B summaries, `a_*` is the first server and `b_*` the second; `p95_change` and `throughput_change` are
fractions, `p95_change_ci` is the 95% bootstrap interval of the P95 change, and `ndcg_delta` is null for N=1, which
has no ranking.
