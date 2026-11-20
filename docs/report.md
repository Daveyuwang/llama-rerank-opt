# Faster `/v1/rerank` in llama.cpp for causal rerankers: design and measurements

This report covers the five patches in `patches/`, made against llama.cpp
[b94959a](https://github.com/ggml-org/llama.cpp/commit/b94959a64d011915d77a5381c7d4846dcda99530). Each table
names the llama.cpp base it was measured on. Most development measurements ran on the earlier base e4b620a; the
patches were then rebased to b94959a without conflicts and with the same added and removed lines, and the key
measurements were repeated there.

## 1. Background

llama-server scores a rerank request with a causal reranker such as Qwen3-Reranker by building one prompt per
document from the model's rerank template (instruction, query, document, closing text), evaluating each prompt as its
own sequence, and reading the score from the last token (RANK pooling). All prompts of a request start with the same
template text and query. On the ToolRet workload below this shared prefix has a median of 83 tokens (apibank) and
92 tokens (metatool), 73 of them template text.

With `-np 32 -kvu`, the 32 prompts of a request occupy 32 slots in one unified KV cache, and the server packs their
tokens into ubatches of up to `-ub` tokens. One ubatch therefore holds parts of many sequences, and the attention
mask is block diagonal: each token sees only its own sequence. This costs time in two places:

- The shared prefix is evaluated once per document, 32 times per request. Evaluating it once would save at most
  41.3% of the prompt tokens of an N=32 request on this workload (70.5% for metatool, which has short documents,
  27.7% for apibank).
- The CUDA flash-attention (FA) kernel computes K/V blocks that the mask hides entirely, namely the blocks of the
  other sequences in the ubatch. Stock trims only the masked columns at the end of each query tile (`KV_max`).
  Nsight Systems measured 34.2 ms of FA kernel time per N=32 request; the visible blocks alone would need about 2 ms.

## 2. Changes

| Patch | Files | Lines | Applies after |
| --- | --- | --- | --- |
| 0001 share the query prefix in a request | `tools/server/server-context.*`, `server-decision.*` | +15/-4 | base |
| 0002 tokenize the query once | `tools/server/server-common.*`, `server-context.cpp` | +67/-1 | 0001 |
| 0003 skip fully masked K/V blocks | `ggml-cuda/fattn-common.cuh`, `fattn-mma-f16.cuh`, `tests/test-backend-ops.cpp` | +302/-35 | base |
| 0004 whole tiles on sm_120 for short K/V | `ggml-cuda/fattn-common.cuh` | +4/-1 | base |
| 0005 waiting sharing groups take slots first (optional) | `tools/server/server-context.cpp`, `server-queue.*` | +144/-10 | 0001 |

Of the 0003 lines, +109/-6 are tests and +18/-6 are in the MMA kernel file.

### 2.1 Share the query prefix in a request (0001)

llama-server already has parent and child tasks for prompts that share a prefix; they were added for `/v1/systemone`
([ggml-org/llama.cpp#29818](https://github.com/ggml-org/llama.cpp/pull/29818)). The parent evaluates the common
prefix, copies its KV cells to the children's sequences, and each child continues from there. 0001 sends
`/v1/rerank` through this path when the model uses causal attention, the context has a memory module (a KV cache to
copy), and the request has no multimodal input:

- `handle_rerank_impl` calls the existing `server_decision_group_tasks`, which groups up to `n_parallel` consecutive
  tasks. Every child keeps at least one token of its own, which RANK pooling needs, so identical documents also work.
- Rerank tasks are stateless, and llama-server does not let stateless tasks reuse a cached prefix
  ([#29556](https://github.com/ggml-org/llama.cpp/pull/29556)). 0001 makes one exception: a child continues from the
  prefix that its own parent has just copied to it, and from nothing else.
- `server_context_meta` gets `can_share_prefix` (causal attention and a memory module), so the request handler can
  read it while the server sleeps.
- The prefix costs an extra decode step, about 13 ms of fixed cost under WSL2. A group therefore shares only if it
  saves at least one ubatch of tokens: `(tasks in group - 1) * n_shared >= n_ubatch`. Without this threshold an
  earlier version made the N=8 P95 latency 8.9% worse.

Non-causal (BERT-style) rerankers keep the original path: their prefix states depend on the document.

### 2.2 Tokenize the query once (0002)

Stock tokenizes the full prompt, template and query included, for every document. 0002 splits the template at the
start of the line that contains `{document}`, tokenizes the part before it (template and query) once per request,
tokenizes the rest per document, and concatenates the tokens. The result equals tokenizing the whole prompt because
the BPE pre-tokenizer always splits between a newline and the text after it, BPE merges stay inside pre-tokens, and special
tokens are split out first. This requires the `{document}` line to start with template text; if it starts with
whitespace or with `{document}` itself, a document starting with a newline could merge with the line break, so such
templates keep the old path. The code also compares the split result with a whole-prompt tokenization for the first
document and falls back on any mismatch. It runs only for text requests and BPE vocabularies.

For N=32 this cuts tokenization from 6.1 to 4.3 ms per request (the rest is the documents themselves), about 2% of
the latency. Scores of 240 held-out requests were bit-identical to 0001 alone.

### 2.3 Skip fully masked K/V blocks in multi-sequence ubatches (0003)

The mask scan kernel `flash_attn_mask_to_KV_max` already computes, per query tile, where the visible columns end
(`KV_max`). 0003 extends the scan and leaves the FA main loop untouched.

**Scan.** For each query tile the scan also finds the first visible 256-column block (`KV_min`) and the longest run
of fully masked blocks between visible ones (the hole). It checks inner blocks in three steps, cheapest first: one
value per block (first row of the tile, first column of the block), which already confirms every visible block of a
causal sequence in one memory round trip; then each warp reads the first row of the blocks still unconfirmed; then a
full check of the blocks whose first row is all `-inf`. Each step only marks blocks visible, so the result equals
the full check alone. The visibility bitmap (one bit per block) lives in the scan kernel's shared memory. The scan
writes two numbers per tile: `origin`, where the block order starts (after the hole, or at `KV_min`, whichever skips
more), and `pend`, the length of that order up to its last visible block.

**FA kernel.** Only the call site changes. Stock assigns blocks to CUDA blocks with stream-k and rotates each tile's
block order by the CUDA block's start, so that neighbouring CUDA blocks share K/V in L2. For a tile with blocks to
skip, 0003 drops the rotation, starts the block order at `origin`, and ends each CUDA block's range at `pend`. The
masked blocks end up at the back of the order and are cut off, the same mechanism stock uses to stop at `KV_max`.
`process_tile` and the K/V loop are unchanged. Tiles without blocks to skip, such as a single causal sequence, keep
the stock order and produce bit-identical results.

**Scope.** Only one hole per tile is skipped. In rerank batches (a shared prefix plus each document) and ordinary
multi-sequence batches a tile has at most one inner masked run; with more, the longest is skipped and the others are
computed as before. The search runs only where stock rotates: the MMA kernel with multi-stage K/V loading (GQA models
on Ampere or newer, `async_kv_preload` on the host side), one K/V slice, and at least 256 queries. Everywhere else
(no GQA, Turing, the vec and tile kernels, decode and small batches, several slices) the scan does only the stock
work and writes zeros. With the threshold at 1024 queries the last, partly filled ubatch of each rerank request was
no longer skipped and throughput dropped from +63.9% to +58.9%; at 256 it recovered and also covers `-ub 512`.

**Tests.** New test-backend-ops cases put several sequences of unequal length into one K/V slice, optionally with a
shared prefix and with positions before the prefix that no query sees (so that the first visible block is not block
0 and both block orders are used); GQA ratios 2, 4 and 8 plus a set without GQA, which must not skip; several
sequences in one kv_rows list; and long single causal sequences (4k, 8k, 32k) for performance. Existing case names
are unchanged.

### 2.4 Whole tiles on sm_120 when K/V is short (0004)

With whole tiles each CUDA block computes complete output tiles and the hardware schedules them; stream-k instead
splits the work evenly by position before knowing which blocks the mask skips. Stock uses whole tiles for scanned
masks only on GB10. On the RTX 5080 whole tiles are faster when the K/V length is close to the query count (the
first ubatches of a prompt, multi-sequence ubatches) and slower on long contexts, where the last wave of tiles is
sparse. 0004 enables whole tiles on `GGML_CUDA_CC_BLACKWELL` when `n_kv <= 2 * n_q`; other GPUs are unchanged.

### 2.5 Waiting sharing groups take slots first (0005, optional)

A sharing group from 0001 starts only when all of its slots are free at once. Under overload with mixed request
sizes, small requests keep taking each slot as it frees, and a request with at least as many documents as slots
waits until all slots happen to be free together (section 4.5). 0005 lets a waiting rerank group go first:

- While a group waits, other new tasks are deferred; the group sits at the front of the deferred queue and is retried
  whenever a slot frees. Groups start in arrival order.
- When any slot runs a task other than rerank or embedding (for example a long completion), 0005 does not hold
  tasks back, so the server is never blocked behind a group that cannot start soon.
- Deadlock guards: a deferred task that finds enough free slots for the waiting group pulls the group back by id; a
  cancelled group is looked up by id only, so the cancel race cannot reorder other tasks; cancelling also removes the
  task from the deferred queue. Slot save, restore and erase tasks remember their slot and run first when it frees.

It changes the scheduling order for all tasks while a group waits, which is a policy decision for the server's
maintainers, so it is a separate patch.

## 3. Method

**Workload.** 40 ToolRet queries ([mangopy/ToolRet-Queries](https://huggingface.co/datasets/mangopy/ToolRet-Queries),
20 metatool and 20 apibank), each with 32 candidates: the labelled positives plus the most similar other tools of the
domain by word overlap, in a fixed shuffled order. Requests use N = 1, 8 or 32 of them. `scripts/rerank_bench.py
prepare` builds the file; queries alternate between a dev and a held-out half in each domain. The configuration
was chosen on dev; every number below is from the held-out half (20 queries).

**Model and server.** Qwen3-Reranker-0.6B Q8_0 (ggml-org GGUF, revision a02f48b), `-fa on -ngl 99 -c 32768 -ctk f16
-ctv f16`, and for the benchmark `-np 32 -kvu -b 8192 -ub 2048` (rerank forces `n_batch = n_ubatch`, so the batch is
2048). On stock, `-ub 2048` is 1.5x faster than `-ub 512`, and split KV is 2.3x slower than unified KV.

**End-to-end statistics.** Builds alternate in each session; 2 sessions of 5 rounds; each round sends every
held-out query at every N once, one request at a time, in a shuffled order, so each build and N has 200 requests.
The P95 change gets a 95% bootstrap interval that resamples queries (all rounds of a query together). Throughput is
requests per second of this sequential sending. Quality compares each candidate's score, the top-1 candidate, and
nDCG@10 with stock. Stock itself drifts by up to 0.058 in score when the batch composition changes.

**Kernels.** `test-backend-ops perf` on FA shapes of Qwen3-0.6B (head dim 128, 16 Q heads, 8 KV heads); each case
runs all builds back to back with the order rotated every repetition. Whole-model prefill uses `llama-bench` with the
same rotation. The RTX 5080 runs long kernels near its 340 W limit at about 2.7 GHz, single samples swing by about
5%, and with 6 repetitions differences below 2% cannot be resolved; the long-context cost was measured with 24.

**Environment.** RTX 5080 16 GB (sm_120), Windows 11 driver 617.42, WSL2 (kernel 6.18), CUDA 13.0. Under WSL2 a
kernel launch costs about 10 us of host time, so launch-bound phases are slower than on native Linux.

## 4. Results

### 4.1 End to end on b94959a

All four patches (0001 to 0004) against stock, held-out, `results/ab-holdout/stock-vs-all.json`:

| N | Scope | P50 stock -> patched | P95 change (95% CI) | Throughput | Top-1 agreement |
| --- | --- | --- | --- | --- | --- |
| 32 | all | 174.9 -> 90.7 ms | -28.1% (-32.3%, -21.8%) | +60.8% | 100% |
| 32 | metatool | 89.2 -> 40.6 ms | -67.0% (-71.5%, -39.5%) | +128.4% | 100% |
| 32 | apibank | 185.5 -> 133.8 ms | -30.0% (-34.0%, -21.2%) | +39.6% | 100% |
| 8 | all | 39.1 -> 33.7 ms | -13.5% (-20.7%, -5.8%) | +13.2% | 95% |
| 1 | all | 12.4 -> 12.7 ms | -4.5% (-17.4%, +11.4%) | +0.7% | 100% |

At N=32 the largest score difference is 0.060 (stock's own drift: 0.058) and nDCG@10 changes by +0.0003. At N=1
there is nothing to share and the scores are identical. The 95% top-1 agreement at N=8 comes from one query whose
candidates all score about 1e-4: the top two swap (1.122e-4 and 1.091e-4 in stock).

0001 alone against stock, in a separate session (`stock-vs-0001.json`): N=32 P50 184.6 -> 106.7 ms, P95 -19.7%
(-21.0%, -10.3%), throughput +41.5%, and the server processed 33.7% fewer prompt tokens (910,050 -> 603,181). At
N=8 the change is not significant: the groups of 8 save about 7 x 88 tokens, below the 2048-token threshold.

Three earlier rounds on e4b620a with the same code paths gave, for all four patches at N=32, P95 -24.6% to -29.8%
and throughput +55.8% to +65.2%; their confidence intervals overlap with each other and with the b94959a round.

### 4.2 Contribution of each part (e4b620a)

Final round on e4b620a, same protocol; "server" is 0001, "CUDA" is 0003 and 0004 (`results/e4b620a/`):

| Comparison, N=32 | P50 | P95 change (95% CI) | Throughput |
| --- | --- | --- | --- |
| stock -> server | 165.3 -> 100.0 ms | -16.5% (-20.5%, -8.6%) | +39.0% |
| stock -> CUDA | 165.3 -> 137.1 ms | -17.0% (-21.1%, -14.3%) | +19.7% |
| server -> all | 100.0 -> 89.0 ms | -9.6% (-14.8%, -3.7%) | +12.1% |
| stock -> all | 165.3 -> 89.0 ms | -24.6% (-29.4%, -16.6%) | +55.8% |

With the CUDA patches alone, 95% of the score differences are at most 0.0001.

### 4.3 Default settings (e4b620a)

Maintainers and users often start with defaults, so the A/B was repeated in two more configurations
(`results/e4b620a/ub512-*.json`, `defaults-*.json`).

| Configuration | Comparison | N=8 throughput | N=8 P95 (95% CI) | N=32 throughput | N=32 P95 (95% CI) |
| --- | --- | --- | --- | --- | --- |
| `-np 32 -kvu`, `-ub 512` | stock -> all | +24.7% | -18.4% (-22.0%, -9.1%) | +58.0% | -27.8% (-30.3%, -22.5%) |
| `-np 32 -kvu`, `-ub 512` | stock -> server | +26.2% | -14.0% (-26.7%, +1.7%) | +50.8% | -20.6% (-25.1%, -14.6%) |
| `-np 32 -kvu`, `-ub 512` | stock -> CUDA | +1.2% | +2.3% (-6.2%, +12.9%) | +4.7% | -2.5% (-4.4%, +1.4%) |
| server defaults (4 slots) | stock -> all | +5.3% | -3.4% (-16.6%, +5.6%) | +5.6% | -1.5% (-9.8%, +4.1%) |

- With `-ub 512` the sharing threshold drops to 512 tokens, N=8 requests share as well, and the server processes
  41.1% fewer tokens. The CUDA patches add little here: kv_rows already packs each 512-token ubatch down to the rows
  of a few sequences, so less work is masked.
- With 4 slots a group has at most 3 children and must save 512 tokens, so a prefix needs at least 171 tokens;
  only 4.9% of the tokens are saved and no change is significant. The large gains need at least as many slots as
  documents per request, which is the configuration a rerank service would use anyway: on the same base, stock's
  N=32 P50 is 286 ms with defaults and 165 to 171 ms with `-np 32 -ub 2048`.

### 4.4 Open loop (b94959a)

Poisson arrivals of held-out N=32 requests, 40 s per rate, latency including queueing (`scripts/openloop.sh`,
`results/openloop/`):

| Arrival rate (/s) | 2 | 4 | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- | --- | --- |
| stock completed (/s) | 1.94 | 4.53 | 5.73 | 7.55 | 7.50 | 7.28 |
| stock P50 / P95 | 177 / 383 ms | 227 / 698 ms | 290 / 866 ms | 3.8 / 6.6 s | 4.7 / 11.8 s | 15.0 / 29.5 s |
| patched completed (/s) | 1.95 | 4.59 | 5.74 | 8.75 | 9.67 | 12.26 |
| patched P50 / P95 | 106 / 174 ms | 124 / 223 ms | 125 / 268 ms | 144 / 482 ms | 169 / 613 ms | 424 ms / 4.8 s |

Stock saturates at about 7.5 requests per second. With the patches the P95 is 0.61 s at 10 per second, and at 12
per second all requests complete but queueing has started (P95 4.8 s). Near saturation the results vary between
runs: an earlier run on e4b620a gave similar numbers up to 10 per second, but at 12 per second the patched server
completed only 10.9 per second (P50 1.0 s, P95 11.9 s). "Completed" counts requests finished in the 40 s, so at low
load it can exceed the nominal rate by chance.

### 4.5 Mixed request sizes and 0005

N=32 requests at 2 per second while N=8 requests arrive at the given rate (`scripts/mixed_load.sh`, e4b620a,
`results/e4b620a/mixed/`):

| N=8 rate | stock N=8 P50 / P95 | stock N=32 P50 / P95 | patched N=8 P50 / P95 | patched N=32 P50 / P95 |
| --- | --- | --- | --- | --- |
| 10/s | 62 / 272 ms | 258 / 724 ms | 45 / 99 ms | 133 / 269 ms |
| 20/s | 218 / 830 ms | 412 / 2109 ms | 59 / 147 ms | 188 / 563 ms |
| 30/s | 3.1 / 9.4 s | 5.6 / 10.8 s | 137 / 460 ms | 2.4 / 12.3 s |
| 40/s | 16.7 / 37.6 s | 23.0 / 38.6 s | 4.7 / 10.5 s | 28.6 / 51.2 s |

Up to 20 per second the patched server is faster for both sizes. At 30 per second stock is already overloaded
(23.5 N=8 requests completed per second); the patched server keeps N=8 fast, but the N=32 requests wait for 32 free
slots at once and their P95 reaches 12.3 s, against about 90 ms of processing.

Two ways of splitting a waiting group were measured and rejected: starting at once with the free slots, and
splitting after waiting 300 or 1000 ms. Both evaluate the prefix again for every split-off document, which makes
throughput worse exactly when the server is loaded (for example, an N=32-only open loop at 12 per second completed
8.6 to 9.5 requests per second, against 10.6 to 10.8 without splitting). 0005 keeps groups whole and lets a waiting
group go first.

0005 at 30 per second, e4b620a, same session (`results/e4b620a/mixed/reserve-*`): N=32 P50 / P95 2.7 / 12.8 s ->
0.21 / 0.94 s, N=8 147 / 459 ms -> 117 / 789 ms. On b94959a, four runs (`results/mixed/`):

| Run | 0001-0004: N=8 P95 | 0001-0004: N=32 P95 | with 0005: N=8 P95 | with 0005: N=32 P95 |
| --- | --- | --- | --- | --- |
| 1 | 0.66 s | 23.1 s | 1.72 s | 1.16 s |
| 2 | 0.70 s | 22.7 s | 1.76 s | 1.22 s |
| 3 | 0.49 s | 14.9 s | 2.01 s | 1.25 s |
| 4 | 0.69 s | 25.9 s | 1.99 s | 1.18 s |

0005 removes the starvation of large requests in every run; the cost for small requests varies with run conditions:
the e4b620a builds measured in run 4 also gave 2.07 s with 0005 against 0.61 s without, so the larger cost on
b94959a does not come from the rebase. Small requests now queue behind large requests that arrived earlier. 0005
passes the same functional checks; with sequential requests its scores are bit-identical to 0001-0004.

### 4.6 FA kernel

Change of the FA time (scan, kernel and stream-k fixup) with 0003 and 0004 against stock (b94959a,
`scripts/fa_kernel_ab.sh`, rotated order, 6 repetitions; median and bootstrap 95% CI of the median;
`results/kernel/`). The stock build carries the test part of 0003, so both builds run the same cases:

| K/V x queries | Mask | Stock (us) | Change |
| --- | --- | --- | --- |
| 2048 x 2048 | 9 sequences | 327.0 | -68.7% [-69.4, -67.3] |
| 2304 x 2048 | 9 sequences after a 96-token shared prefix | 375.4 | -60.6% [-62.1, -57.6] |
| 2048 x 2048 | 1 causal sequence | 326.2 | -27.5% [-29.7, -24.3] |
| 4096 x 2048 | 1 causal sequence | 615.0 | -12.3% [-13.3, -10.3] |
| 4096 x 1024 | 1 causal sequence | 333.5 | -0.7% [-2.4, +3.1] |
| 8192 x 2048 | 1 causal sequence | 1185.3 | +2.1% [-0.5, +3.7] |
| 8192 x 1024 | 1 causal sequence | 622.3 | -1.0% [-2.6, +1.3] |
| 32768 x 2048 | 1 causal sequence | 4455.7 | +0.9% [-0.8, +1.9] |
| 32768 x 1024 | 1 causal sequence | 2274.0 | +2.2% [-0.1, +5.1] |

Single causal sequences with K/V up to twice the query count gain from 0004; longer ones are untouched by 0004 and
show only the cost of 0003, which 6 repetitions cannot resolve. That cost was measured on e4b620a with 24 rotated
repetitions, the published 0003 against stock, plus two measurement-only builds that each contain half of the
change:

| K/V x queries | 0003 | New scan only | New call site only |
| --- | --- | --- | --- |
| 32768 x 2048 | +0.84% (+0.75, +0.92), slower in 24 of 24 | +0.19% (+0.11, +0.34) | +0.66% (+0.58, +0.71) |
| 8192 x 2048 | +1.00% (+0.76, +1.12), slower in 24 of 24 | +0.32% (+0.03, +0.48) | +0.56% (+0.43, +0.89) |

The two halves add up. The scan's share matches the 4 to 7 us that Nsight shows for the longer scan. The call site
adds a few integer operations and loads per tile; `ptxas -v` reports the same registers and spills as stock for this
kernel (255 registers, 76 bytes spill stores, 180 bytes spill loads) and the PTX main loop is unchanged, so the
0.6% most likely comes from instruction scheduling or code layout; comparing SASS would confirm it. In whole-model
prefill this cost appears as a 0.4% slowdown at 32k tokens (next section).

### 4.7 Whole-model prefill (b94959a)

`llama-bench`, single sequence, `-ub 2048 -b 2048`, 8 rounds with rotated order, median [min, max] of the per-round
changes against stock (`results/prefill/llama-bench-ub2048.jsonl`):

| Prompt | Stock (t/s) | 0004 alone | 0001 to 0004 |
| --- | --- | --- | --- |
| 512 | 42749 | -0.6% [-7.5, +7.0] | -2.2% [-4.5, +5.6] |
| 1024 | 61408 | +3.9% [+1.4, +4.9] | +3.7% [-0.8, +5.0] |
| 2048 | 59728 | +8.0% [+4.1, +9.4] | +7.7% [+5.3, +9.2] |
| 4096 | 54076 | +7.7% [+6.6, +8.1] | +6.6% [+6.2, +7.6] |
| 8192 | 45705 | +2.8% [+2.1, +4.1] | +2.4% [+1.2, +3.6] |
| 16384 | 34076 | +1.0% [+0.8, +1.4] | +0.3% [+0.0, +0.5] |
| 32768 | 22477 | +0.3% [+0.0, +0.6] | -0.4% [-0.5, -0.3] |

With the default `-ub 512` (`llama-bench-ub512.jsonl`) the medians lie between -5.7% and +0.9% with wide per-round
ranges, so no change is measurable: only the first two ubatches of a prompt satisfy `n_kv <= 2 * n_q`. 0004 helps
prefill with large ubatches. At 16k and 32k tokens the full set is 0.7% behind 0004 alone in every round, which
is the cost of 0003 on long single sequences.

### 4.8 Where the remaining time goes (e4b620a)

With 0001 and 0003, an N=32 request takes 87 to 92 ms. Temporary timers in a local build (40 held-out requests sent
one at a time) split it as follows:

| Part | Per request |
| --- | --- |
| Host time of kernel launches (`graph_compute`) | 26.9 ms |
| Waiting for the GPU after the launches | 44.8 ms |
| Mask and input preparation (`set_inputs`) | 1.3 ms |
| Graph build, KV slot search, batch checks | 2.1 ms |
| Server work before and after decode | 2.2 ms |
| Outside the decode loop (HTTP, tokenization, scheduling) | about 9.6 ms |

Each decode launches about 750 kernels at about 10 us each under WSL2, 7.5 to 8.4 ms per decode regardless of its
size; even an 80-token prefix decode takes about 13 ms. CUDA graphs do not help because ggml-cuda reuses a graph only
when consecutive graphs have the same properties, and every rerank ubatch has a different token count and K/V
length. Operator fusion or reusing graphs per shape bucket would be the next steps; both are separate topics.

## 5. Design notes

**Why all skipping happens before the main loop.** The first version kept a bitmap of visible blocks and looked it
up inside the K/V loop. It worked, but made single-sequence prefill 3.6% to 7.4% slower at kernel level. The MMA
kernel already uses all 255 registers and spills; in a controlled experiment every extra value kept alive from
one loop iteration to the next added about 0.8% (0 values: +0.0% to +0.8%, 2: +1.2% to +2.0%, 3: +2.2% to +4.1%). A copy of the
stock loop for tiles without holes did not help either (+2.9% to +4.8%), because registers are allocated per
function. The final design therefore decides everything in the scan and at the call site, and the loop is the stock
loop.

**Why tiles with holes are not rotated.** Shifting the block order to start after the hole was not enough: stock
rotates each CUDA block's range again for L2 sharing, which put the hole in the middle of the ranges where it cannot
be cut. A small Python model of the stream-k split reproduced the measured busiest-block counts (196 blocks for stock
and for the shifted order, 49 with the hole skipped) and showed that tiles with blocks to skip must not rotate.

**Stream-k imbalance in stock.** The same model shows that stock is unbalanced on a single causal sequence: at
2048 x 2048 the busiest CUDA block computes 196 blocks while the mean is 110, because stream-k divides positions
evenly and positions past `KV_max` cost nothing. Balancing by visible work could be up to about 40% faster. 0004 gets
most of this on these shapes by letting the hardware schedule whole tiles (-30% at 2048 x 2048); a stream-k split by
visible work would also need changes to the fixup and was not attempted.

**Scan cost.** Checking each inner block with a full-block test that needs two `__syncthreads` raised the scan time
at 32k context from 7.5 to 150 us. Warp-parallel first-row reads brought it to 25 us, and the one-value-per-block
first step to about 15 us (11 to 12 us at 8k), while multi-sequence scans stay at about 6.5 us.

## 6. Correctness

- **FA numerics.** All `FLASH_ATTN_EXT` cases of test-backend-ops (4,462, compared with the CPU backend) pass on a
  fresh build of b94959a with the patches, including the new multi-sequence, shared-prefix and leading-hidden-position cases and the kv_rows cases with
  several sequences in one list (256 and 512 queries, head dims 128 and 256).
- **The tests reach the skipping paths.** Three deliberately injected errors were each caught by the new cases:
  ending every CUDA block one K/V unit early (16 of 32 new cases fail), shortening the order after the hole by one
  256-column block (12 fail, all prefix cases) and shortening the order from `KV_min` by one block (15 fail). The
  cases without GQA pass under all three, as they should, since that kernel does not skip.
- **Upstream server unit tests.** `test-server-component` passes on stock and patched b94959a (57 tests, 257
  assertions).
- **Rerank behaviour.** Upstream's `test_rerank.py` downloads a non-causal model, so `scripts/server_checks.py`
  replays its assertions (ranking, TEI format, HTTP 400 on invalid input, usage, `top_n`, no idle-slot caching) on
  Qwen3-Reranker, plus cases that exercise sharing: identical and empty documents, 64 documents (more than the
  slots), children spanning several ubatches (`-ub 64`), split KV, and 8 concurrent requests. In four
  configurations (1 slot; 4 slots unified; 4 slots split; 32 slots) all assertions hold, and status codes and token
  usage equal stock's.
- **Single sequence unchanged.** With one slot (no sharing, one sequence per ubatch) the 38 responses are
  byte-identical to stock.
- **Concurrency.** Stock ranks 7 or 8 of 8 requests differently when they are sent concurrently instead of one at a
  time (largest score difference 0.0035 to 0.13). The patched server changes at most 2, both near-ties (largest
  difference 0.0003).
- **Other GPUs.** The 40 FA translation units compile without warnings for sm_75, sm_80, sm_89 and sm_90; they were
  not run there. HIP and MUSA were not compiled.
- **Multi-stream decode.** `llama-batched-bench` with 4 and 8 streams (4096-token prompts, 128 generated tokens)
  shows +2.0% and 0.0% token generation, in the noise; prompt processing in the same test is 6.7% to 8.1% faster
  (from 0004).

## 7. Limitations

- One GPU (RTX 5080, sm_120), WSL2, one model (Qwen3-Reranker-0.6B Q8_0). WSL2 inflates launch overhead and absolute
  latencies. Register pressure differs on other architectures, so the single-sequence results need re-measuring
  there.
- Scores are not bit-identical to stock: 0001 changes how sequences are batched (largest difference 0.061, the size
  of stock's own drift), and 0003 changes the accumulation order for tiles with skipped blocks (numerically
  equivalent, the FA tests pass against the CPU).
- 0003 covers the MMA path with multi-stage K/V loading only; models without GQA and Turing GPUs run the stock code.
  The vec, tile and WMMA paths and HIP/MUSA were not validated. Long single sequences pay about 0.8% to 1.0% in FA
  kernel time; whole-model prefill of 32k tokens is 0.4% slower than stock.
- 0001 needs at least as many slots as documents per request for large gains; the `n_ubatch` threshold was tuned
  where an extra decode costs about 13 ms, and could be lower on native Linux. Reusing one prefix for all groups
  of a request when slots are few would need further scheduling changes.
- Under overload with mixed request sizes, large requests starve without 0005, and 0005 raises the tail latency of
  small requests by an amount that varies between runs.
- The 0004 rule (`n_kv <= 2 * n_q`) comes from three context lengths on one GPU.

## 8. Reproducing the numbers

| Result | Script | Data in this repository |
| --- | --- | --- |
| 4.1 end to end | `scripts/ab_holdout.sh` | `results/ab-holdout/` (summaries and pooled raw runs) |
| 4.2, 4.3 per part, defaults | `scripts/ab_holdout.sh` with `SERVER_ARGS` | `results/e4b620a/*.json` |
| 4.4 open loop | `scripts/openloop.sh` | `results/openloop/` |
| 4.5 mixed sizes, 0005 | `scripts/mixed_load.sh` | `results/mixed/`, `results/e4b620a/mixed/` |
| 4.6 FA kernel | `scripts/fa_kernel_ab.sh` | `results/kernel/` |
| 4.7 prefill | `scripts/prefill_bench.sh` | `results/prefill/` |
| 6 rerank behaviour | `scripts/server_checks.py` | |

`python3 scripts/ab_compare.py results/ab-holdout/runs/stock-a.jsonl results/ab-holdout/runs/all.jsonl` recomputes
the table in 4.1 from the recorded runs without a GPU.
