# llama-rerank-opt

Patches for [llama.cpp](https://github.com/ggml-org/llama.cpp) that make `/v1/rerank` faster with causal (LLM-based)
rerankers such as Qwen3-Reranker, together with the benchmark scripts and the recorded data behind the numbers.

With Qwen3-Reranker-0.6B on an RTX 5080 and 32 server slots, requests of one query and 32 documents run with 61%
higher throughput and a 28% lower P95 latency, and every request keeps the same top-ranked document.

## What the patches change

A causal reranker scores each document as its own prompt, and all prompts of a request begin with the same template
text and query. Stock llama-server evaluates this shared prefix once per document. When many short prompts share an
ubatch, its CUDA flash-attention (FA) kernel also computes K/V blocks that the attention mask hides completely.

| Patch | Change |
| --- | --- |
| `0001-server-rerank-shared-prefix` | Evaluates the query prefix once per group of documents and copies its KV cells to the rest, using llama-server's parent and child tasks from [#29818](https://github.com/ggml-org/llama.cpp/pull/29818). Only when a group saves at least one ubatch of tokens. |
| `0002-server-rerank-tokenize-query-once` | Tokenizes the template and query once per request instead of once per document. Applies after 0001. |
| `0003-cuda-fa-skip-masked-kv` | The FA mask scan also finds, per query tile, the first visible K/V block and the longest fully masked run; the MMA kernel skips them through its block order, with an unchanged main loop. Adds test-backend-ops cases. |
| `0004-cuda-fa-whole-tiles-sm120` | On sm_120 GPUs, schedules whole tiles instead of stream-k when the K/V length is at most twice the query count. |
| `0005-server-rerank-group-slot-reservation` | Optional. Lets a waiting rerank group take free slots before newer tasks, so that large requests do not starve under overload. Applies after 0001. |

The patches apply to llama.cpp [b94959a](https://github.com/ggml-org/llama.cpp/commit/b94959a64d011915d77a5381c7d4846dcda99530).
They are not part of upstream llama.cpp. [docs/report.md](docs/report.md) explains the design and all measurements.

## Results

Qwen3-Reranker-0.6B Q8_0, RTX 5080 under WSL2, llama.cpp b94959a. Each request holds one ToolRet tool-retrieval
query and N candidate tools; the numbers come from the held-out half of the queries, with
`llama-server -np 32 -kvu -b 8192 -ub 2048 -fa on` and 200 requests per build and N, sent one at a time:

| N=32 | stock | patches 0001 to 0004 | change |
| --- | --- | --- | --- |
| P50 latency | 174.9 ms | 90.7 ms | -48.2% |
| P95 latency | 212.3 ms | 152.6 ms | -28.1% (95% CI -32.3% to -21.8%) |
| Throughput | | | +60.8% |
| Top-1 document same as stock | | | 100% |

- Three earlier rounds on the base e4b620a gave P95 -24.6% to -29.8% and throughput +55.8% to +65.2%.
- 0001 alone at N=32: P95 -19.7%, throughput +41.5%, 33.7% fewer prompt tokens.
- All four patches at N=8: P95 -13.5%, throughput +13.2%.
- FA kernel time with 0003 and 0004 (test-backend-ops, Qwen3-0.6B attention shapes, 2048 queries): 9 sequences
  -68.7%, 9 sequences after a 96-token shared prefix -60.6%, one causal 2048-token sequence -27.5%. Long single
  sequences pay about 1% (0.8% to 1.0% over 24 repetitions).
- Single-sequence prefill (`llama-bench`, `-ub 2048`): +3.7% to +7.7% for 1k to 4k prompts; 32k prompts are 0.4%
  slower.
- Open loop with Poisson arrivals of N=32 requests: stock completes at most about 7.5 requests per second. With the
  patches the P95 latency is 0.61 s at 10 requests per second, and at 12 per second all requests still complete
  (P95 4.8 s).
- Correctness: all 4,462 FA cases of test-backend-ops pass, upstream's `test-server-component` passes, and the
  assertions of upstream's rerank tests hold in four server configurations.

## When it helps

- Prefix sharing needs at least as many slots as documents per request, such as `-np 32` for 32 documents. With
  llama-server's default 4 slots it seldom triggers (about +5% throughput, not significant).
- The default `-ub 512` with `-np 32` also works: +58% throughput and P95 -28% at N=32 (measured on e4b620a).
- 0003 skips blocks for GQA models on Ampere or newer GPUs in ubatches of at least 256 queries; everything else runs
  the stock code.
- 0004 changes only sm_120 GPUs and speeds up prefill with large ubatches.
- Under overload with a mix of small and large requests, a large request waits until all its slots are free at once.
  0005 removes that starvation, at a cost in small-request tail latency.
- Everything was measured on one GPU (RTX 5080) under WSL2 with one model.

## Try it

You need Linux or WSL2, an NVIDIA GPU with the CUDA toolkit, CMake, git, curl, Python 3 (standard library only) and,
for the model download, the `hf` command from `huggingface_hub`. From this repository:

```sh
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout b94959a64d011915d77a5381c7d4846dcda99530
cp -r llama.cpp llama.cpp-stock
scripts/apply_patches.sh llama.cpp     # 0001 to 0004; add patches/0005-*.patch for the slot reservation
scripts/build.sh llama.cpp             # llama-server, llama-bench and test-backend-ops in llama.cpp/build/bin
scripts/build.sh llama.cpp-stock
```

`build.sh` targets RTX 50 GPUs; set `CUDA_ARCH` for others (for example `CUDA_ARCH=89` for RTX 40) and pass more
cmake options in `CMAKE_ARGS`. If linking fails because CUDA or OpenSSL live outside the system library path (for
example in a conda environment), `CMAKE_ARGS=-DBUILD_SHARED_LIBS=OFF` links llama.cpp statically. The llama.cpp
configure step downloads the server's web UI by default; `-DLLAMA_USE_PREBUILT_UI=OFF` builds without it, which the
benchmarks do not need.

Download the model and build the workload:

```sh
hf download ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF qwen3-reranker-0.6b-q8_0.gguf \
    --revision a02f48bb4f057028298c21fa033da2b30d7742d5 --local-dir models
python3 scripts/rerank_bench.py prepare --out data/cases.jsonl
```

`prepare` reads ToolRet from the Hugging Face datasets server. The workload of the results has the SHA256
`89ef79e89aab13efa35c5c2922dd758a1d78d7cd596221b169aa71c7bbb6fd33`; a different hash means ToolRet has changed.

Serve with the patched build:

```sh
llama.cpp/build/bin/llama-server -m models/qwen3-reranker-0.6b-q8_0.gguf --reranking -fa on -ngl 99 \
    -np 32 -kvu -b 8192 -ub 2048 -c 32768
```

Compare stock and patched in one short session (about 20 seconds on the RTX 5080):

```sh
export MODEL=models/qwen3-reranker-0.6b-q8_0.gguf
SESSIONS=1 ROUNDS=1 scripts/ab_holdout.sh runs/quick \
    stock=llama.cpp-stock/build/bin/llama-server patched=llama.cpp/build/bin/llama-server
```

For each N and domain it prints P50 and P95 in ms for stock (A) and patched (B), the P95 change with a 95% bootstrap
interval over queries, the throughput change, the largest and the P95 score difference, top-1 agreement and the
nDCG@10 change. Without `SESSIONS` and `ROUNDS` it runs the full protocol of the results (2 sessions of 5 rounds,
about 2 minutes).

Without a GPU, the recorded runs give the results table again:

```sh
python3 scripts/ab_compare.py results/ab-holdout/runs/stock-a.jsonl results/ab-holdout/runs/all.jsonl
```

## Reproducing the other measurements

The server scripts need `MODEL` and take builds as `NAME=PATH`, the first one being the reference.

| Script | Measures | Recorded output |
| --- | --- | --- |
| `scripts/ab_holdout.sh` | held-out A/B; `SERVER_ARGS` sets another server configuration | `results/ab-holdout/`, `results/e4b620a/` |
| `scripts/openloop.sh` | Poisson arrivals of N=32 requests at fixed rates | `results/openloop/` |
| `scripts/mixed_load.sh` | N=8 and N=32 requests arriving together | `results/mixed/` |
| `scripts/fa_kernel_ab.sh` | FA kernel time with test-backend-ops, rotated build order | `results/kernel/` |
| `scripts/prefill_bench.sh` | single-sequence prefill with llama-bench, rotated build order | `results/prefill/` |
| `scripts/server_checks.py` | `/rerank` behaviour against stock: upstream test assertions, sharing edge cases, concurrency | |

The FA correctness tests run with `llama.cpp/build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0`. For the
kernel A/B the stock build needs the new test cases, so before building it run
`git -C llama.cpp-stock apply --include='tests/*' patches/0003-cuda-fa-skip-masked-kv.patch`.

## Layout

| Path | Content |
| --- | --- |
| `patches/` | the five patches, in application order |
| `scripts/` | apply, build, workload, benchmarks and checks |
| `results/` | recorded measurements, described in [results/README.md](results/README.md) |
| `docs/report.md` | design, method, results and limitations |

## License

MIT, see [LICENSE](LICENSE). The patches modify llama.cpp, which is MIT licensed by the ggml authors.
