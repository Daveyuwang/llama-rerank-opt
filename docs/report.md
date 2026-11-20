# Faster `/v1/rerank` in llama.cpp for causal rerankers: design and measurements

This report covers the four patches in `patches/`, made against llama.cpp
[b94959a](https://github.com/ggml-org/llama.cpp/commit/b94959a64d011915d77a5381c7d4846dcda99530). All measurements
below ran on that base with these patches.

## 1. Background

llama-server scores a rerank request with a causal reranker such as Qwen3-Reranker by building one prompt per
document from the model's rerank template (instruction, query, document, closing text), evaluating each prompt as its
own sequence, and reading the score from the last token (RANK pooling). All prompts of a request start with the same
template text and query. On the ToolRet workload below this shared prefix has a median of 83 tokens (apibank) and
92 tokens (metatool), 73 of them template text.

With `-np 32 -kvu`, the 32 prompts of a request occupy 32 slots in one unified KV cache, and the server packs their
tokens into ubatches of up to `-ub` tokens. One ubatch therefore holds parts of many sequences, and the attention
mask is block diagonal: each token sees only its own sequence. This costs time in two places:

- The shared prefix is evaluated once per document, 32 times per request. Evaluating it once saves 42.3% of the
  prompt tokens of the N=32 requests in this workload (section 4.1).
- The CUDA flash-attention (FA) kernel computes K/V blocks that the mask hides entirely, namely the blocks of the
  other sequences in the ubatch. Stock trims only the masked columns at the end of each query tile (`KV_max`). A
  profile of stock on the development base e4b620a showed 34.2 ms of FA kernel time per N=32 request, while the
  visible blocks alone need about 2 ms.

## 2. Changes

| Patch | Files | Code lines | Test lines | Applies after |
| --- | --- | --- | --- | --- |
| 0001 share the query prefix in a request | `tools/server/server-context.*`, `server-decision.*` | +20/-4 | +121 | base |
| 0002 tokenize the query once | `tools/server/server-common.*`, `server-context.cpp` | +65/-1 | +18 | 0001 |
| 0003 skip fully masked K/V blocks | `ggml-cuda/fattn-common.cuh`, `fattn-mma-f16.cuh` | +185/-29 | +110/-6 | base |
| 0004 whole tiles on sm_120 for short K/V | `ggml-cuda/fattn-common.cuh` | +3/-1 | | base |

The test lines of 0001 and 0002 are in `tools/server/tests/unit/test_rerank.py`, those of 0003 in
`tests/test-backend-ops.cpp`. A fifth patch reserved free slots for waiting sharing groups (section 4.7); a review
found that it could stall other requests, so it was dropped.

### 2.1 Share the query prefix in a request (0001)

llama-server already has parent and child tasks for prompts that share a prefix; they were added for `/v1/systemone`
([ggml-org/llama.cpp#29818](https://github.com/ggml-org/llama.cpp/pull/29818)). The parent evaluates the common
prefix, copies its KV cells to the children's sequences, and each child continues from there. 0001 sends
`/v1/rerank` through this path when the model uses causal attention, the context has a memory module and a unified
KV cache, and the request has no media files:

- `handle_rerank_impl` calls the existing `server_decision_group_tasks`, which groups up to `n_parallel` consecutive
  tasks. Every child keeps at least one token of its own, which RANK pooling needs, so identical documents work too.
- Rerank tasks are stateless, and llama-server does not let stateless tasks reuse a cached prefix
  ([#29556](https://github.com/ggml-org/llama.cpp/pull/29556)). 0001 makes one exception: a child continues from the
  prefix that its own parent copied to it, and from nothing else.
- `server_context_meta` gets `can_share_prefix` and the context's ubatch size, so the request handler can read both
  while the server sleeps.
- The prefix costs an extra decode, about 13 ms of fixed cost under WSL2. A group therefore shares only if it saves
  at least one ubatch of tokens: `(tasks in group - 1) * n_shared >= n_ubatch`, with the ubatch size the context
  uses. Without this threshold an earlier version made the N=8 P95 latency 8.9% worse.
- With split KV (one KV stream per slot), copying the prefix to a child copies the parent's whole stream in every
  layer. An earlier version shared there too and made N=32 requests slower than stock (section 4.5), so 0001
  requires a unified KV cache.

The new tests compare each document's score in a shared request with its score when sent alone (in logit space,
with the tolerance that batching needs in stock as well, section 6), and check in `/metrics` that fewer prompt
tokens were evaluated: with 8 and with 3 slots, with identical documents, with split KV (no sharing), with
concurrent requests, below the threshold, and with the prompt cache enabled.

Non-causal (BERT-style) rerankers keep the original path: their prefix states depend on the document.

### 2.2 Tokenize the query once (0002)

Stock tokenizes the full prompt, template and query included, for every document. 0002 splits the template at the
start of the line that contains `{document}`, tokenizes the part before it (template and query) once per request,
tokenizes the rest per document, and concatenates the tokens. The result equals tokenizing the whole prompt: the BPE
pre-tokenizer splits between a line break and the text after it, BPE merges stay inside pre-tokens, and special
tokens are split out first. This requires the `{document}` line to start with template text; if it starts with
whitespace or with `{document}` itself, a document that starts with a line break could merge with the one before
it, so such templates keep the stock path. The tokens of the first document are also compared with those of
`format_prompt_rerank`, the stock function, and any mismatch falls back to it. It runs only for text requests and
BPE vocabularies.

The test sends documents that start or end with line breaks or spaces, an empty document, special-token text and
template placeholders in the query and the documents, with one slot so that nothing is shared, and requires
bit-identical scores and equal token counts compared with requests of one document. During development 0002 cut
tokenization from 6.1 to 4.3 ms per N=32 request (the rest is the documents themselves), about 2% of the latency.

### 2.3 Skip fully masked K/V blocks in multi-sequence ubatches (0003)

The mask scan kernel `flash_attn_mask_to_KV_max` already computes, per query tile, where the visible columns end
(`KV_max`). 0003 extends the scan and leaves the FA main loop untouched.

**Scan.** For each query tile the scan also finds the first visible 256-column block (`KV_min`) and the longest run
of fully masked blocks between visible ones (the hole). It checks inner blocks in three steps, cheapest first: one
value per block (first row of the tile, first column of the block), which already confirms every visible block of a
causal sequence in one memory round trip; then the first row of the blocks still unconfirmed, several blocks per
warp; then a full check of the blocks whose first row is all `-inf`. Each step only marks blocks visible, so the
result equals the full check alone. The visibility bitmap (one bit per block) lives in the scan kernel's shared
memory. The scan writes two numbers per tile: `origin`, where the block order starts (after the hole, or at
`KV_min`, whichever skips more), and `pend`, the length of that order up to its last visible block.

**FA kernel.** Only the call site changes. Stock assigns blocks to CUDA blocks with stream-k and rotates each tile's
block order by the CUDA block's start, so that neighbouring CUDA blocks share K/V in L2. For a tile with blocks to
skip, 0003 drops the rotation, starts the block order at `origin`, and ends each CUDA block's range at `pend`. The
masked blocks end up at the back of the order and are cut off, the same mechanism stock uses to stop at `KV_max`.
`process_tile` and the K/V loop are unchanged. Tiles without blocks to skip, such as a single causal sequence, keep
the stock order and produce bit-identical results.

**Scope.** Only one hole per tile is skipped. In rerank batches (a shared prefix plus each document) and ordinary
multi-sequence batches a tile has at most one inner masked run; with more, the longest is skipped and the others are
computed as before. The search runs only where stock rotates: the MMA kernel with multi-stage K/V loading (GQA
models on Ampere or newer, `async_kv_preload` on the host side), one K/V slice, and at least 256 queries. Everywhere
else (no GQA, Turing, the vec and tile kernels, decode and small batches, several slices) the scan does only the
stock work and writes zeros. With a threshold of 1024 queries the last, partly filled ubatch of each rerank request
was no longer skipped, and throughput dropped from +63.9% to +58.9% on the dev half; at 256 it recovered and also
covers `-ub 512`.

**Tests.** New test-backend-ops cases put several sequences of unequal length into one K/V slice, optionally with a
shared prefix and with positions before the prefix that no query sees (so that the first visible block is not block
0 and both block orders are used); GQA ratios 2, 4 and 8 plus a set without GQA, which must not skip; K/V lengths of
12k and 20k, so that the bitmap spans several words; several sequences in one kv_rows list; and long single causal
sequences (4k, 8k, 32k) for performance. Existing case names are unchanged.

### 2.4 Whole tiles on sm_120 when K/V is short (0004)

With whole tiles each CUDA block computes complete output tiles and the hardware schedules them; stream-k instead
splits the work evenly by position before the mask scan cuts off the blocks past `KV_max`. Stock uses whole tiles
for scanned masks only on GB10. On the RTX 5080 whole tiles are faster when the K/V length is close to the query
count (the first ubatches of a prompt, multi-sequence ubatches) and slower on long contexts, where the last wave of
tiles is sparse. 0004 enables whole tiles on `GGML_CUDA_CC_BLACKWELL` when `n_kv <= 2 * n_q`; other GPUs are
unchanged.

## 3. Method

**Workload.** 40 ToolRet queries ([mangopy/ToolRet-Queries](https://huggingface.co/datasets/mangopy/ToolRet-Queries),
20 metatool and 20 apibank), each with 32 candidates: the labelled positives plus the most similar other tools of the
domain by word overlap, in a fixed shuffled order. Requests use N = 1, 8 or 32 of them. `scripts/rerank_bench.py
prepare` builds the file; queries alternate between a dev and a held-out half in each domain. The configuration was
chosen on dev; every number below is from the held-out half (20 queries).

**Servers.** llama-server runs Qwen3-Reranker-0.6B Q8_0 (ggml-org GGUF, revision a02f48b) with `-fa on -ngl 99 -c
32768 -ctk f16 -ctv f16` and, unless noted, `-np 32 -kvu -b 8192 -ub 2048` (rerank forces `n_batch = n_ubatch`, so
the batch is 2048). vLLM 0.34.0 (PyTorch 2.14, CUDA 13.2) serves the original bf16 checkpoint (Qwen/Qwen3-Reranker-
0.6B, revision e61197e) as a yes/no classifier, as vLLM's documentation describes, with its defaults (automatic
prefix caching, chunked prefill, CUDA graphs), `--max-model-len 8192` and `--gpu-memory-utilization 0.8`. For vLLM
the client applies the rerank template that llama-server reads from the GGUF file, so both servers score the same
prompt text and count the same tokens. The 4B model is Qwen/Qwen3-Reranker-4B (revision 22e6836), converted to
Q8_0 with llama.cpp's `convert_hf_to_gguf.py` at b94959a for llama-server and served in bf16 by vLLM.

**End-to-end statistics.** Each session runs every server once, and the order of the servers rotates from session
to session; 2 sessions of 5 rounds, each after a discarded warm-up round. A round sends every held-out query at
every N once, one request at a time, in a shuffled order, so each server and N has 200 requests. The P95 change gets
a 95% bootstrap interval that resamples queries (all rounds of a query together). Throughput is requests per second
of this sequential sending. Quality compares each candidate's score, the top-1 candidate and nDCG@10 with stock.
Stock itself changes scores with the batch composition: the same 8 documents sent together instead of one at a time
move by up to 0.24 in logit space (0.019 in score).

**Kernels.** `test-backend-ops perf` on FA shapes of Qwen3-0.6B (head dim 128, 16 Q heads, 8 KV heads); each case
runs all builds back to back with the order rotated every repetition. Whole-model prefill uses `llama-bench` with the
same rotation. The RTX 5080 runs long kernels near its 340 W limit at about 2.7 GHz, single samples swing by about
5%, and with 6 repetitions differences below 2% cannot be resolved.

**Environment.** RTX 5080 16 GB (sm_120), Windows 11 driver 617.42, WSL2 (kernel 6.18), CUDA 13.0 for llama.cpp.
Under WSL2 a kernel launch costs about 10 us of host time, so launch-bound phases are slower than on native Linux.

## 4. Results

### 4.1 End to end

All four patches against stock, held-out, `results/ab-holdout/ab-stock-all.json`:

| N | Scope | P50 stock -> patched | P95 change (95% CI) | Throughput | Top-1 agreement |
| --- | --- | --- | --- | --- | --- |
| 32 | all | 190.6 -> 90.3 ms | -33.8% (-35.9%, -26.3%) | +68.2% | 100% |
| 32 | metatool | 95.0 -> 42.8 ms | -70.4% (-72.1%, -47.4%) | +134.3% | 100% |
| 32 | apibank | 197.9 -> 138.0 ms | -34.5% (-37.1%, -24.8%) | +46.8% | 100% |
| 8 | all | 41.5 -> 35.0 ms | -18.8% (-22.3%, -15.3%) | +17.5% | 95% |
| 1 | all | 13.7 -> 13.9 ms | +2.1% (-8.8%, +11.7%) | -1.8% | 100% |

At N=32 the largest score difference is 0.060 and nDCG@10 changes by +0.0003. At N=1 there is nothing to share and
the scores are identical. The 95% top-1 agreement at N=8 comes from one query whose candidates all score about 1e-4,
so that its top two swap; the CUDA patches alone and vLLM swap them too.

With 0001 the server evaluates 42.3% fewer prompt tokens at N=32 (141,550 -> 81,720 for one round of the 20
held-out queries, `results/ab-holdout/prompt-tokens-n32.txt`). Over all sizes of the full run the reduction is 33.3%,
because groups of 8 save less than one ubatch and do not share.

### 4.2 Contribution of each part

Same sessions; "server" is 0001 and 0002, "CUDA" is 0003 and 0004 (`results/ab-holdout/`):

| N=32 | P50 | P95 change (95% CI) | Throughput |
| --- | --- | --- | --- |
| stock -> server | 190.6 -> 105.4 ms | -21.9% (-24.5%, -13.3%) | +45.9% |
| stock -> CUDA | 190.6 -> 151.8 ms | -18.0% (-23.6%, -17.9%) | +23.2% |
| stock -> all | 190.6 -> 90.3 ms | -33.8% (-35.9%, -26.3%) | +68.2% |

At N=8 the server patches give +2.9% throughput (not significant) and the CUDA patches +14.0%. With the CUDA patches
alone, 95% of the score differences at N=32 are at most 0.0001.

### 4.3 Against vLLM

Same sessions, P50 / P95 in ms (`results/ab-holdout/ab-stock-vllm.json`, `ab-all-vllm.json`):

| N | stock | patched | vLLM |
| --- | --- | --- | --- |
| 1 | 13.7 / 27.4 | 13.9 / 28.0 | 6.3 / 16.1 |
| 8 | 41.5 / 61.0 | 35.0 / 49.6 | 19.8 / 27.4 |
| 32 | 190.6 / 227.1 | 90.3 / 150.3 | 41.3 / 50.9 |

vLLM serves this workload about twice as fast as patched llama.cpp: at N=32 its throughput is 121% higher than
patched and 271% higher than stock. These two figures leave out one vLLM request that stalled for 34.6 s, with no
warning in vLLM's log; with it, vLLM's summed time at N=32 exceeds stock's. vLLM is also faster at N=1, where
nothing is shared: it replays CUDA graphs, while llama.cpp launches about 750 kernels per decode at about 10 us
each under WSL2.

Most of vLLM's lead comes from its prefix cache. In a separate session without it (`--no-enable-prefix-caching`,
`results/configs/vllm-noapc-stock-vllm-noapc.json`), vLLM's N=32 P50 was 138.4 ms against 185.9 ms for stock in
that session, slower than patched llama.cpp. Scores agree: at N=32 the top-1 document matches stock for every
request, the largest score difference is 0.26 (bf16 against Q8_0 weights) and nDCG@10 changes by -0.003.

### 4.4 Qwen3-Reranker-4B

Same protocol with the 4B model (`results/model-4b/`):

| N=32 | stock | patched | vLLM |
| --- | --- | --- | --- |
| P50 | 612.4 ms | 259.0 ms | 269.6 ms |
| P95 | 737.8 ms | 462.2 ms | 639.4 ms |
| Throughput against stock | | +83.0% | +40.2% |

The patches gain more with the larger model (+83.0% against +68.2%, P95 -37.4% with CI -38.1% to -30.2%), because
the prefix and the masked blocks cost more compute per token. At N=32 patched llama.cpp now completes requests 30%
faster than vLLM; at N=8 vLLM is faster (P50 53.5 against 120.8 ms, 37% higher throughput). With this model the
work is compute-bound, and llama.cpp's int8 kernels for Q8_0 weights do less work than vLLM's bf16 ones, so this
comparison mixes the weight format with the frameworks.

### 4.5 Other server configurations

All four patches against stock (`results/configs/`):

| Configuration | stock N=32 P50 | N=8 throughput | N=32 throughput | N=32 P95 (95% CI) |
| --- | --- | --- | --- | --- |
| `-np 32 -kvu -ub 2048` (section 4.1) | 190.6 ms | +17.5% | +68.2% | -33.8% (-35.9%, -26.3%) |
| `-np 32 -kvu -ub 512` | 255.8 ms | +21.0% | +56.0% | -27.8% (-30.1%, -22.8%) |
| server defaults (4 slots, unified KV, `-ub 512`) | 259.7 ms | +12.7% | +12.4% | -11.4% (-18.8%, -8.2%) |
| `-np 32 -ub 2048`, split KV | 445.8 ms | +3.1% | +7.1% | -6.0% (-8.8%, -2.5%) |

- With `-ub 512` the sharing threshold drops to 512 tokens, so groups of 8 share as well.
- With 4 slots a group has at most 3 children and must save 512 tokens, so a prefix needs at least 171 tokens;
  this workload's prefixes are shorter, and the gain comes from the CUDA patches.
- With split KV, which llama-server uses whenever `-np` is set without `-kvu`, stock is about 2.3 times slower at
  N=32 than with a unified cache. Before 0001 required a unified cache, sharing made N=32 requests slower there:
  throughput -5.0%, P95 +4.3% (CI +1.1%, +9.2%) for the server patches against stock
  (`results/configs/splitkv-sharing-stock-server.json`). Now the patched builds evaluate the same prompt tokens as
  stock and return the same scores. The server patches alone still show N=32 throughput -1.7% and P95 +6.1% (CI
  -1.9%, +10.2%), and an N=8 P95 of +11.8% (CI +0.9%, +21.6%); they run the stock path here, so these differences
  are variation between sessions in this slow configuration, which the query bootstrap does not cover.

### 4.6 Open loop

Poisson arrivals of held-out N=32 requests, 40 s per rate, latency including queueing (`scripts/openloop.sh`,
`results/openloop/`). "Completed" is the number of successful requests divided by the time until the last response,
so under overload it approaches the capacity:

| Arrival rate (/s) | 2 | 4 | 6 | 8 | 10 | 12 |
| --- | --- | --- | --- | --- | --- | --- |
| stock completed (/s) | 1.94 | 4.54 | 5.73 | 7.69 | 7.63 | 7.41 |
| stock P50 / P95 | 178 / 380 ms | 221 / 694 ms | 282 / 819 ms | 3.3 / 6.0 s | 4.2 / 11.0 s | 14.4 / 28.3 s |
| patched completed (/s) | 1.95 | 4.59 | 5.74 | 8.75 | 9.70 | 12.09 |
| patched P50 / P95 | 108 / 165 ms | 121 / 206 ms | 126 / 267 ms | 140 / 440 ms | 149 / 546 ms | 506 ms / 5.9 s |

Stock saturates at about 7.6 requests per second. With the patches the P95 is 0.55 s at 10 per second, and at 12
per second all requests complete but queueing has started (P95 5.9 s). The arrivals are random, so at low rates the
completed rate can exceed the nominal one.

### 4.7 Mixed request sizes

N=32 requests at 2 per second while N=8 requests arrive at the given rate (`scripts/mixed_load.sh`,
`results/mixed/`), P50 / P95:

| N=8 rate | stock N=8 | stock N=32 | patched N=8 | patched N=32 |
| --- | --- | --- | --- | --- |
| 10/s | 58 / 239 ms | 258 / 701 ms | 41 / 91 ms | 117 / 289 ms |
| 20/s | 105 / 625 ms | 0.37 / 3.5 s | 55 / 137 ms | 170 / 548 ms |
| 30/s | 1.7 / 6.1 s | 3.2 / 7.1 s | 81 / 279 ms | 0.33 / 2.2 s |
| 40/s | 13.7 / 30.0 s | 18.7 / 30.7 s | 1.8 / 3.9 s | 20.9 / 43.4 s |

Up to 30 per second the patched server is faster for both sizes. At 40 per second stock is overloaded for both
sizes (23.6 N=8 requests completed per second), while the patched server still completes 37.7 N=8 requests per
second, but its N=32 requests wait for 32 free slots at once and their P95 reaches 43 s. The wait varies between
runs: four earlier runs with the previous version of the patches reached a P95 of 15 to 26 s already at 30
per second.

Three ways around this starvation were tried. Splitting a waiting group, at once or after 300 or 1000 ms, evaluates
the prefix again for every split-off document and made throughput worse exactly when the server was loaded (an
N=32-only open loop at 12 per second completed 8.6 to 9.5 requests per second, against 10.6 to 10.8 without
splitting). A fifth patch let a waiting group take free slots before newer tasks; it removed the starvation (N=32
P95 at 30 per second from 15 to 26 s down to about 1.2 s) but raised the N=8 P95 from about 0.6 s to 1.7 to 2.0 s,
and a code review found that it could stall deferred requests while a long task ran, start groups out of arrival
order, and leave a request waiting behind a deferred slot save. It was dropped. The fix that remains open is to
start a group on the slots that are free and keep its prefix in the KV cache for the children that start later
(section 7).

### 4.8 FA kernel

Change of the FA time (scan, kernel and stream-k fixup) with 0003 and 0004 against stock
(`scripts/fa_kernel_ab.sh`, rotated order, 6 repetitions; median and bootstrap 95% CI of the median;
`results/kernel/`). The stock build carries the test part of 0003, so both builds run the same cases:

| K/V x queries | Mask | Stock (us) | Change |
| --- | --- | --- | --- |
| 2048 x 2048 | 9 sequences | 315.5 | -68.7% [-69.2, -67.9] |
| 2304 x 2048 | 9 sequences after a 96-token shared prefix | 359.8 | -59.5% [-60.8, -59.1] |
| 2048 x 2048 | 1 causal sequence | 317.0 | -26.3% [-26.9, -25.7] |
| 4096 x 2048 | 1 causal sequence | 602.0 | -11.6% [-12.4, -11.0] |
| 4096 x 1024 | 1 causal sequence | 320.2 | +0.9% [-0.1, +1.3] |
| 8192 x 2048 | 1 causal sequence | 1153.8 | +0.8% [+0.0, +1.7] |
| 8192 x 1024 | 1 causal sequence | 607.8 | +0.5% [+0.0, +1.3] |
| 32768 x 2048 | 1 causal sequence | 4362.8 | +0.7% [+0.5, +1.0] |
| 32768 x 1024 | 1 causal sequence | 2235.5 | +0.4% [+0.1, +0.8] |

Multi-sequence ubatches gain from 0003; single causal sequences whose K/V length is at most twice the query count
gain from 0004. Longer single sequences, which neither patch speeds up, show the cost of 0003's search: +0.4% to
+0.9%. During development, 24 rotated repetitions with builds that each contained half of 0003 attributed about a
quarter of this cost to the longer scan and the rest to the call site. `ptxas -v` reports the same registers and
spills for the kernel as stock (255 registers, 76 bytes of spill stores, 180 bytes of spill loads) and the PTX main
loop is unchanged, so the call-site share most likely comes from instruction scheduling or code layout.

### 4.9 Whole-model prefill

`llama-bench`, single sequence, `-ub 2048 -b 2048`, 8 rounds with rotated order, median [min, max] of the per-round
changes against stock (`results/prefill/prefill-ub2048.txt`):

| Prompt | Stock (t/s) | 0004 alone | 0001 to 0004 |
| --- | --- | --- | --- |
| 512 | 44025 | -1.6% [-3.9, +6.2] | -1.3% [-5.4, +0.2] |
| 1024 | 63916 | +4.5% [+1.8, +5.7] | +4.8% [+1.9, +5.0] |
| 2048 | 62747 | +7.6% [+6.3, +8.3] | +7.6% [+4.0, +8.4] |
| 4096 | 57046 | +7.3% [+6.7, +7.5] | +6.8% [+5.2, +7.5] |
| 8192 | 48115 | +2.9% [+2.6, +3.2] | +2.2% [+1.5, +2.6] |
| 16384 | 35949 | +1.1% [+0.9, +1.5] | +0.4% [+0.3, +0.9] |
| 32768 | 23608 | +0.4% [+0.2, +0.5] | -0.3% [-0.4, -0.2] |

0004 speeds up prefill with large ubatches, most for prompts of 1k to 4k tokens. At 8k tokens and more the full set
falls behind 0004 alone by 0.7%, which is the cost of 0003's search on long single sequences; at 32k it is 0.3%
slower than stock. With the default `-ub 512` (`prefill-ub512.txt`) the medians lie between -4.2% and +2.0% with
wide per-round ranges, so no change is measurable: only the first two ubatches of a prompt satisfy
`n_kv <= 2 * n_q`.

## 5. Design notes

**Why all skipping happens before the main loop.** The first version kept a bitmap of visible blocks and looked it
up inside the K/V loop. It worked, but made single-sequence prefill 3.6% to 7.4% slower at kernel level. The MMA
kernel already uses all 255 registers and spills; in a controlled experiment every extra value kept alive from one
loop iteration to the next added about 0.8%. A copy of the stock loop for tiles without holes did not help either
(+2.9% to +4.8%), because registers are allocated per function. The final design therefore decides everything in
the scan and at the call site, and the loop is the stock loop.

**Why tiles with holes are not rotated.** Shifting the block order to start after the hole was not enough: stock
rotates each CUDA block's range again for L2 sharing, which put the hole in the middle of the ranges where it cannot
be cut. A small Python model of the stream-k split reproduced the measured busiest-block counts (196 blocks for stock
and for the shifted order, 49 with the hole skipped) and showed that tiles with blocks to skip must not rotate.

**Stream-k imbalance in stock.** The same model shows that stock is unbalanced on a single causal sequence: at
2048 x 2048 the busiest CUDA block computes 196 blocks while the mean is 110, because stream-k divides positions
evenly and positions past `KV_max` cost nothing. 0004 gets most of a balanced split on these shapes by letting the
hardware schedule whole tiles; a stream-k split by visible work would also need changes to the fixup.

**Scan cost.** Checking each inner block with a full-block test that needs two `__syncthreads` raised the scan time
at 32k context from 7.5 to 150 us. Reading the first rows warp-parallel brought it to 25 us, and the
one-value-per-block first step to about 15 us (11 to 12 us at 8k), while multi-sequence scans stay at about 6.5 us.

## 6. Correctness

- **FA numerics.** All 4,464 `FLASH_ATTN_EXT` cases of test-backend-ops, compared with the CPU backend, pass on a
  build of b94959a with the patches, including the new multi-sequence, shared-prefix, leading-hidden-position and
  multi-word cases and the kv_rows cases with several sequences in one list (`results/tests/`).
- **The tests reach the skipping paths.** During development three deliberately injected errors were each caught
  by the new cases: ending every CUDA block one K/V unit early, shortening the order after the hole by one block,
  and shortening the order from `KV_min` by one block. The cases without GQA passed under all three, as they should,
  since that kernel does not skip.
- **Server tests.** The 11 new tests in `test_rerank.py` pass on the patched build. On stock, the 3 tests that
  expect sharing fail on the token count, as they should, and the others pass. They load Qwen3-Reranker-0.6B, so
  they are marked `slow`; upstream's other rerank tests use a non-causal model, which never shares.
- **Scores.** Sharing changes how sequences are batched, which changes rounding: the largest score difference at
  N=32 is 0.060, while the same documents batched differently in stock move by up to 0.24 in logit space. With one
  slot (no sharing) 0002's scores are bit-identical to requests of one document.
- **Other GPUs.** During development the FA translation units compiled without warnings for sm_75, sm_80, sm_89 and
  sm_90; they were not run there. HIP and MUSA compile the same files with `find_masked` always false and were not
  built.

## 7. Limitations and future work

- One GPU (RTX 5080, sm_120) under WSL2 and two model sizes. WSL2 inflates launch overhead and absolute latencies.
  Register pressure differs on other architectures, so the single-sequence costs of 0003 need measuring there.
- A sharing group starts only when all its slots are free (section 4.7). Starting it on the free slots and keeping
  its prefix for later children (a pinned parent slot, or a spare sequence id under `-kvu`) would remove the
  starvation without evaluating the prefix again.
- The prefix costs a decode of its own and one copy per child. With a unified cache the prefix tokens could instead
  go into the first document batch once, tagged with the sequence ids of the parent and all children, which
  llama.cpp's batch API supports; this would remove the extra decode and the threshold, so that N=8 requests share
  as well.
- 0003 decides from shapes alone whether to search for holes, so every single-sequence prefill ubatch of at least
  256 queries runs a search that finds nothing; long single sequences pay 0.8% to 1.0% in FA kernel time. A hint
  from the graph that the K/V view holds several sequences, as the sparse FA path already receives, and a kernel
  instance without the new call site would remove this cost. The scan also runs once per layer although all layers
  share the mask, as stock's does.
- 0003 covers the MMA path with multi-stage K/V loading only; models without GQA and Turing GPUs run the stock code.
  Ragged per-sequence K/V slices would make the hole search unnecessary on every backend, but need a new layout for
  the FA op.
- The 0004 rule (`n_kv <= 2 * n_q`) comes from three context lengths on one GPU. Stream-k is unbalanced on other
  architectures too; dividing it by visible work would replace the rule.
- vLLM is about twice as fast with the small model (section 4.3). Most of the remaining llama.cpp time is kernel
  launches under WSL2 (about 750 per decode); CUDA graphs per shape bucket or operator fusion would be the next
  steps.

## 8. Reproducing the numbers

| Result | Script | Data in this repository |
| --- | --- | --- |
| 4.1 to 4.3 end to end, parts, vLLM | `scripts/ab_holdout.sh` | `results/ab-holdout/` (summaries and pooled raw runs) |
| 4.3 vLLM without prefix caching, 4.5 other configurations | `scripts/ab_holdout.sh` with `VLLM_ARGS`, `SERVER_ARGS` | `results/configs/` |
| 4.4 4B model | `scripts/ab_holdout.sh` with `MODEL` | `results/model-4b/` |
| 4.6 open loop | `scripts/openloop.sh` | `results/openloop/` |
| 4.7 mixed sizes | `scripts/mixed_load.sh` | `results/mixed/` |
| 4.8 FA kernel | `scripts/fa_kernel_ab.sh` | `results/kernel/` |
| 4.9 prefill | `scripts/prefill_bench.sh` | `results/prefill/` |

`python3 scripts/ab_compare.py results/ab-holdout/runs/stock.jsonl results/ab-holdout/runs/all.jsonl` recomputes
the table in 4.1 from the recorded runs without a GPU.
