# llama-rerank-opt

Patches for [llama.cpp](https://github.com/ggml-org/llama.cpp) that make `/v1/rerank` faster with causal (LLM-based)
rerankers such as Qwen3-Reranker, together with the benchmark scripts and the recorded data behind the numbers.

With Qwen3-Reranker-0.6B on an RTX 5080 and 32 server slots, requests of one query and 32 documents run with 68%
higher throughput and a 34% lower P95 latency, and every request keeps the same top-ranked document. With
Qwen3-Reranker-4B the throughput rises by 83%, which puts llama.cpp ahead of vLLM on the same GPU for that model.

## What the patches change

A causal reranker scores each document as its own prompt, and all prompts of a request begin with the same template
text and query. Stock llama-server evaluates this shared prefix once per document. Its CUDA flash-attention (FA)
kernel also computes the K/V blocks that the attention mask hides completely, which is most of the work in an
ubatch that holds many short prompts.

| Patch | Change |
| --- | --- |
| `0001-server-rerank-shared-prefix` | Evaluates the query prefix once per group of documents and copies its KV cells to the rest, using llama-server's parent and child tasks from [#29818](https://github.com/ggml-org/llama.cpp/pull/29818). Only with a unified KV cache, and only when a group saves at least one ubatch of tokens. Adds server tests. |
| `0002-server-rerank-tokenize-query-once` | Tokenizes the template and query once per request instead of once per document. Applies after 0001. |
| `0003-cuda-fa-skip-masked-kv` | The FA mask scan also finds, per query tile, the first visible K/V block and the longest fully masked run; the MMA kernel skips them through its block order, with an unchanged main loop. Adds test-backend-ops cases. |
| `0004-cuda-fa-whole-tiles-sm120` | On sm_120 GPUs, schedules whole tiles instead of stream-k when the K/V length is at most twice the query count. |

The patches apply to llama.cpp [b94959a](https://github.com/ggml-org/llama.cpp/commit/b94959a64d011915d77a5381c7d4846dcda99530).
They are not part of upstream llama.cpp. [docs/report.md](docs/report.md) explains the design and all measurements.

## Results

RTX 5080 under WSL2, llama.cpp b94959a. Each request holds one ToolRet tool-retrieval query and N candidate tools;
the numbers come from the held-out half of the queries, with `llama-server -np 32 -kvu -b 8192 -ub 2048 -fa on`
and 200 requests per server and N, sent one at a time. llama.cpp runs Q8_0 GGUF files; vLLM 0.34 runs the original
bf16 checkpoints with its default prefix caching, on the same prompt text.

| Qwen3-Reranker-0.6B, N=32 | stock | patched | change | vLLM |
| --- | --- | --- | --- | --- |
| P50 latency | 190.6 ms | 90.3 ms | -52.6% | 41.3 ms |
| P95 latency | 227.1 ms | 150.3 ms | -33.8% (95% CI -35.9% to -26.3%) | 50.9 ms |
| Throughput | | | +68.2% | +271% over stock |
| Top-1 document same as stock | | | 100% | 100% |

| Qwen3-Reranker-4B, N=32 | stock | patched | change | vLLM |
| --- | --- | --- | --- | --- |
| P50 latency | 612.4 ms | 259.0 ms | -57.7% | 269.6 ms |
| P95 latency | 737.8 ms | 462.2 ms | -37.4% (95% CI -38.1% to -30.2%) | 639.4 ms |
| Throughput | | | +83.0% | +40.2% over stock |

- Server patches alone (0001, 0002) at N=32: throughput +45.9%, P95 -21.9%; the server processes 42.3% fewer
  prompt tokens. CUDA patches alone (0003, 0004): throughput +23.2%, P95 -18.0%.
- At N=8: throughput +17.5%, P95 -18.8%, almost all from the CUDA patches, because a group of 8 documents saves
  less than one 2048-token ubatch and does not share its prefix.
- vLLM's prefix caching gives it most of its lead with the small model: without it, vLLM takes 138 ms at N=32
  (P50), slower than patched llama.cpp. One of 600 vLLM requests stalled for 34.6 s; its throughput figure excludes
  that request. With the 4B model, patched llama.cpp completes N=32 requests 30% faster than vLLM, where int8
  kernels for Q8_0 weights do less work than vLLM's bf16 ones.
- FA kernel time with 0003 and 0004 (test-backend-ops, Qwen3-0.6B attention shapes, 2048 queries): 9 sequences
  -68.7%, 9 sequences after a 96-token shared prefix -59.5%, one causal 2048-token sequence -26.3%. Long single
  sequences pay 0.4% to 0.9%.
- Single-sequence prefill (`llama-bench`, `-ub 2048`): +4.8% to +7.6% for 1k to 4k prompts, from 0004; 32k prompts
  are 0.3% slower.
- Open loop with Poisson arrivals of N=32 requests: stock completes at most about 7.6 requests per second. With the
  patches the P95 latency is 0.55 s at 10 requests per second, and at 12 per second all requests still complete
  (P95 5.9 s).
- Correctness: all 4,464 FA cases of test-backend-ops pass; the new server tests show that shared and separate
  evaluation give the same scores (up to the rounding that batching changes in stock too) and the same token
  counts, and that the query part tokenized once gives bit-identical scores.

## When it helps

- Prefix sharing needs at least as many slots as documents per request (such as `-np 32` for 32 documents) and a
  unified KV cache (`-kvu`). llama-server uses split KV whenever `-np` is set, and then stock itself is about 2.3
  times slower at N=32. With split KV, 0001 does not share, because copying a prefix would copy whole KV streams.
- With `-ub 512`: throughput +56.0% at N=32 and +21.0% at N=8, where groups of 8 now share as well.
- With the server defaults (4 slots, `-ub 512`) a group saves too few tokens to share; the CUDA patches give
  +12% throughput at N=8 and N=32.
- 0003 skips blocks for GQA models on Ampere or newer GPUs in ubatches of at least 256 queries; everything else runs
  the stock code. 0004 changes only sm_120 GPUs and helps prefill with large ubatches.
- A sharing group starts only when all of its slots are free at once. Under overload with a mix of small and large
  requests, small requests keep taking the slots as they free up, and large requests can wait for seconds.
- Measured on one GPU (RTX 5080) under WSL2.

## Try it

You need Linux or WSL2, an NVIDIA GPU with the CUDA toolkit, CMake, git, curl, Python 3 (standard library only) and,
for the model download, the `hf` command from `huggingface_hub`. From this repository:

```sh
git clone https://github.com/ggml-org/llama.cpp
git -C llama.cpp checkout b94959a64d011915d77a5381c7d4846dcda99530
cp -r llama.cpp llama.cpp-stock
scripts/apply_patches.sh llama.cpp     # all four patches
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
`89ef79e89aab13efa35c5c2922dd758a1d78d7cd596221b169aa71c7bbb6fd33`, which every recorded run carries; a different
hash means ToolRet has changed.

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
about 2 minutes per server). To add vLLM, install it, download the Hugging Face checkpoint (for example
`Qwen/Qwen3-Reranker-0.6B`) and add `vllm=vllm:PATH_TO_CHECKPOINT`; `VLLM` names the vllm command.

Without a GPU, the recorded runs give the results table again:

```sh
python3 scripts/ab_compare.py results/ab-holdout/runs/stock.jsonl results/ab-holdout/runs/all.jsonl
```

## Reproducing the other measurements

The server scripts need `MODEL` and take servers as `NAME=PATH`, the first one being the reference.

| Script | Measures | Recorded output |
| --- | --- | --- |
| `scripts/ab_holdout.sh` | held-out A/B; `SERVER_ARGS` sets another server configuration, `vllm:DIR` adds vLLM | `results/ab-holdout/`, `results/configs/`, `results/model-4b/` |
| `scripts/openloop.sh` | Poisson arrivals of N=32 requests at fixed rates | `results/openloop/` |
| `scripts/mixed_load.sh` | N=8 and N=32 requests arriving together | `results/mixed/` |
| `scripts/fa_kernel_ab.sh` | FA kernel time with test-backend-ops, rotated build order | `results/kernel/` |
| `scripts/prefill_bench.sh` | single-sequence prefill with llama-bench, rotated build order | `results/prefill/` |

The FA correctness tests run with `llama.cpp/build/bin/test-backend-ops test -o FLASH_ATTN_EXT -b CUDA0`. For the
kernel A/B the stock build needs the new test cases, so before building it run
`git -C llama.cpp-stock apply --include='tests/*' patches/0003-cuda-fa-skip-masked-kv.patch`. The server tests of
0001 and 0002 are in llama.cpp's `tools/server/tests/unit/test_rerank.py`, marked `slow`; they load the model from
`QWEN3_RERANKER_LOCAL_MODEL` or download it.

## Layout

| Path | Content |
| --- | --- |
| `patches/` | the four patches, in application order |
| `scripts/` | apply, build, workload, benchmarks |
| `results/` | recorded measurements, described in [results/README.md](results/README.md) |
| `docs/report.md` | design, method, results and limitations |

## License

MIT, see [LICENSE](LICENSE). The patches modify llama.cpp, which is MIT licensed by the ggml authors.
