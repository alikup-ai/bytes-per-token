# Phase 3: engine v1 on the board (2026-09-25)

Code: `engine/` (design in `engine/DESIGN.md`). Run: `.venv/Scripts/python engine/test.py <arm> --port COM6 --runs 3`.
Board: ESP32-S3-N16R8, 240 MHz, PSRAM 80 MHz. Greedy decoding, 48 new tokens after a prompt
(positions ~10-60), 5 timing repeats per board run.

## Results
| model | val loss (train, QAT or fp32) | engine NLL / PyTorch int4 NLL (512 val tokens) | argmax agreement | **ms/token, 2 cores** | tok/s | 1 core | board runs | checks |
|---|---|---|---|---|---|---|---|---|
| qat4 (dense SwiGLU, QAT 4-bit) | 2.020 | 1.8135 / 1.8144 | 99.0% | **16.55** | 60.4 | 20.91 | 3 | 12/12 |
| relu (PTQ 4-bit) | 1.999 fp32 | 1.8589 / 1.8617 | 99.2% | **12.86** | 77.8 | 16.73 | 2+2 | 10/10 |
| ple64 (PTQ 4-bit) | 1.961 fp32 | 1.8827 / 1.8835 | 98.8% | 17.16 | 58.3 | 21.62 | 2+2 | 10/10 |
| **combo** (PLE + ReLU + QAT 4-bit) | **2.008** | 1.8758 / 1.8764 | 99.2% | **14.16** | **70.6** | 18.20 | 3 | 12/12 |


Every board run: PIE on 2 cores == PIE on 1 core == scalar C == PC, bit for bit (token digests,
digests of every logit, NLL sum bits). Timing spread across repeats and runs: < 0.1%.

## Stage breakdown, ms/token (2 cores)
| model | input | attn matvecs | attention | FFN | PLE | head |
|---|---|---|---|---|---|---|
| qat4 | 0.01 | 2.63 | 1.60 | 8.77 | — | 3.54 |
| relu | 0.01 | 2.65 | 1.61 | **5.05** | — | 3.54 |
| combo | 0.27 | 2.83 | 1.64 | **4.33** | 1.55 | 3.54 |

## Reading
- ~~The cost model held: predicted 15.9 ms, measured 16.55 (+4%).~~ **Corrected 2026-09-29:** 15.9 ms was the
  prediction at position 128, the measurement ran at positions 13-60. At the same positions (and counting the fp32
  scale per 64 weights) the model predicts 14.95 ms: **10% low** (`python costmodel/ideas.py`, last table). For
  esp32-ai's model it is 12-14% low. Inside the plan's ±15%, but not the "+4%" written here before.
- **ReLU gate-first works on the chip:** FFN 8.77 -> 5.05 ms (x1.74 on the stage, x1.29 overall). It only
  skips `up` so far; sparse `down` is the next step.
- **Combo is the best trade:** better quality than qat4 (2.008 vs 2.020 val loss; with the second seed added
  2026-09-29: 2.0082 / 2.0116 vs 2.0201 / 2.0284, better on both seeds by 0.012 / 0.017) and 17% faster. PLE costs
  1.8 ms/token (0.27 input + 1.55 per-layer), mostly its matvecs; the flash row read is small.
- The **second core** gives x1.26-1.30, in line with the 73 -> 83 MB/s bandwidth gain plus parallel attention.
- Against esp32-ai (10.6 tok/s): not a like-for-like comparison, since its model is 15x larger with a 25k vocabulary.
  Running their model on this engine is the fair test (the cost model predicts ~45 tok/s).

## Accuracy vs PyTorch, and the near-tie rule
int8 activations move logits by a median 0.025-0.03 (p99 0.12-0.19, logit std ~2.9). NLL agrees
within 0.003. Greedy generations match PyTorch token for token except where PyTorch's own top-2 gap
is below that error (seen: 0.001-0.056). There, either token is a legitimate choice. The test fails a
divergence only where PyTorch's margin exceeds the measured p99 gap error.

## Where the time goes next (ms/token of combo, and the fix)
- head 3.54: tied 4096x128 matrix streamed from PSRAM -> two-stage head (post-hoc: 11% of rows, top-1 97%).
- FFN 4.33: `down` still read in full -> sparse down (transposed, PIE axpy).
- attention 1.6 at short context: per-token V^T writes touch 32 PSRAM cache lines per head -> V ring in SRAM.
- scales: fp32 per 64 weights = 6% of bytes -> fp16 or per-row.
- SRAM holds 186 KB of weights; PSRAM 814 KB.

## v2 (2026-09-25): one change at a time, each measured on the board (combo model, 2 cores)
| step | ms/token | tok/s | result |
|---|---|---|---|
| v1 | 14.16 | 70.6 | |
| 1. V^T block staged in SRAM, flushed every 16 positions | 13.77 | 72.6 | bit-identical. Attention barely moved (1.64 -> 1.58): scattered PSRAM writes were **not** the bottleneck (hypothesis refuted). The total fell 0.39 ms, not 0.06; the other 0.33 ms was not attributed to a stage (cache effects of the removed writes on later stages are a guess, not measured) |
| profile: attention sub-steps + microbench | | | own exp 174 cycles, recip 115: the S3 FPU has multi-cycle latency, so dependent-op depth is what costs |
| 2. exp in Estrin order, 3 Newton steps, inlined | 13.59 | 73.6 | max rel. error 2.9e-7 (was ~1e-7). Moves int8 rounding in places: NLL +0.002, inside the int8 noise |
| 3. more weights in SRAM (32 KB reserve instead of a fixed 180 KB budget) | 14.25 -> **13.17** | 75.9 | first try was slower: weights took SRAM before the engine's 32 KB buffer, which fell back to PSRAM. Fixed by allocating engine buffers first |
| 4. fp16 scales (-6% bytes) | 13.78 | — | **slower**: converting 8192 scales/token costs more than the bytes saved. Reverted |
| 5. two-stage head (256 k-means clusters, 15% of rows) | 11.73 | 85.3 | read rows bit-identical to the full head; top-1 same as full head 99.6% |
| 5b. cluster selection by heap instead of repeated scans | **10.64** | 94.0 | head 3.64 -> 1.23 ms |
| 6. sparse `down` (transposed codes, new PIE QACC kernel) | 10.55 | 94.8 | bit-identical; gain eaten by a 1000-cycle unpack (memcpy per lane) |
| 6b. unpack with aligned word loads | **10.21** | **97.9** | FFN 3.87 -> 3.35 ms |

**Bugs found on the way:** (a) the 128-bit PIE stores align their address down to 16 bytes:
a unit test against the C reference caught a wrong QACC lane layout before the engine used it.
(b) The 3 KB model struct lived on the 3.5 KB default main stack and overflowed it into the heap
once the head fields were added. Now static, and the main stack is 8 KB.

**Where 10.2 ms go now:** FFN 3.35, attention matvecs 2.60, attention 1.48, PLE 1.30, head 1.22, input 0.27.

## v2 final check, all models (`engine/out/final_v2.sh`, log `engine/out/final_v2.log`)
| model | v1 tok/s | **v2 tok/s** | ms/token | 1 core | engine NLL vs PyTorch | head top-1 vs full | board runs | checks |
|---|---|---|---|---|---|---|---|---|
| qat4 (SwiGLU) | 60.4 | **80.2** | 12.47 | 16.05 | -0.0006 | 99.4% | 1 | 10/10 |
| relu | 77.8 | **111.6** | 8.96 | 12.14 | -0.0009 | 99.4% | 1 | 10/10 |
| ple64 | 58.3 | **77.0** | 12.99 | 16.66 | +0.0004 | 99.6% | 1 | 10/10 |
| **combo** (PLE + ReLU + QAT4) | 70.6 | **97.9** | 10.21 | 13.55 | +0.0014 | 99.6% | 3 | 14/14 |

Every board run is bit-identical across PIE on 2 cores, PIE on 1 core, scalar C, and the PC.
The fastest model is relu (111.6 tok/s). combo is 12% slower but carries PLE and was trained for 4 bits;
relu here is fp32-trained and rounded to 4 bits afterwards. The QAT relu arm (`relu_qat4`) has
val loss **2.028 / 2.033** (s0 / s1, mean 2.031) at 4 bits, against qat4 (SwiGLU) 2.020 / 2.028 and combo
2.008 / 2.012 (second seeds added 2026-09-29, `train/sweep_review.sh`). So: relu is fastest (~112 tok/s)
at +0.020 loss, combo is the best quality at 98 tok/s.

## v3 (2026-09-26): per-token work (phase-4 finding: per-token compute now limits)
Profiled on the board by kind (core 0, combo, ms/token): matvec incl. waits 6.33, attention 1.48,
**input quantization 0.97** (38 vectors/token, ~6k cycles each), element-wise 0.46 (GELU), head
cluster selection 0.37, norms 0.25, live-list bookkeeping 0.19.

| change | effect |
|---|---|
| quantization: max of \|x\| on IEEE bits (integer compare), round by adding 1.5*2^23, 4-way unrolled; group sums fused | prep 0.97 -> 0.37 ms |
| GELU / SiLU from 1024-interval tables, linear interpolation (max error 2e-5 / 6e-5) | element-wise 0.46 -> 0.20 |
| head: fill non-read logits in token order from a token->cluster byte map, not scattered through the permutation | 0.37 -> 0.26 |
| one live-neuron list per layer, reused by sparse down | 0.19 -> 0.12 |

Rounding is now round-half-to-even and GELU is tabulated, so digests differ from v2. PC == board and
all accuracy checks still hold (combo engine NLL 1.8766 vs PyTorch 1.8764).

| model | v2 tok/s | **v3 tok/s** | full head |
|---|---|---|---|
| combo | 98.5 | **109.5** (9.14 ms) | 85.6 |
| esp32-ai TinyStories | 87.9 | **99.8** (10.02 ms) | 42.5 (4.3x their 9.88) |

Remaining per-token work on combo: ~2.65 ms (attention 1.34, prep 0.37, head selection 0.26, norms 0.25,
element-wise 0.20, bookkeeping 0.12) against ~6.5 ms of weight streaming.

## Attention: how much is PSRAM? (experiment LLM_DBG_WARM_KV, 2026-09-26)
Before each layer's attention, core 0 reads every KV cache line that attention will touch. Attention
then runs on a hot cache. Results are bit-identical. Combo, 2 cores:

| context | attention | attention, hot cache | memory share | cost of the warm-up pass (1 core) |
|---|---|---|---|---|
| generation, pos ~13-60 | 1.32 ms | 0.60 | 0.72 | 0.93 |
| eval, pos 0-255 (avg 128) | 3.53 ms | 1.59 | 1.94 | 2.71 |

- KV traffic is bandwidth, not only latency: at position 128 attention reads ~220 KB/token (K, V,
  scales), as much as all attention weights. A random 64-byte miss (~170 cycles) costs about the same
  as streaming those 64 bytes, so prefetching cannot make the bytes cheaper. It can only overlap them
  with CPU-only work.
- The only idle bus time to overlap with is attention's own hot compute (~1.6 ms at pos 128) plus
  norms and quantization. Estimate for a per-head DMA pipeline (blocked V layout, double buffers per
  core, GDMA at ~68 MB/s shared by both cores): **-0.5 ms at short context, about -1.1 ms at position
  128**, for a layout change, DMA descriptors and ~20-40 KB of SRAM buffers taken from weights.
- Moving KV into SRAM instead of weights is a wash: each byte in SRAM saves one PSRAM read per token either way.

## Batched known tokens (llm_forward_batch, 2026-09-27)
Each phase runs all tokens of a batch back to back, so a matrix (all < 64 KB) comes from the data cache
after the first token. Attention stays in position order. Bit-identical to single steps: the last
logits are the same for batch 1, 2, 4, 8 and 16 on the PC and on the board (scalar and PIE).
64 known tokens, ms/token, 2 cores:

| model, PSRAM | n=1 | 2 | 4 | 8 | 16 | gain at 16 |
|---|---|---|---|---|---|---|
| combo, 80 MHz | 9.04 | 6.98 | 5.93 | 5.42 | 5.21 | x1.74 |
| combo, 120 MHz | 7.33 | 6.01 | 5.32 | 5.00 | 4.87 | x1.51 |
| esp32-ai, 80 MHz | 9.94 | 6.53 | 4.80 | 3.96 | 3.53 | x2.82 |
| esp32-ai, 120 MHz | 7.95 | 5.49 | 4.25 | 3.64 | 3.34 | x2.38 |

The gain is below the W/C estimate (combo ideal ~2.7 ms at n=16), so cache reuse is partial. Row-tiling the
matrices over the batch would guarantee reuse. Prompts are now prefilled in batches of 16.

## PSRAM 120 MHz (experimental, see docs/phase1-bench.md), 2 cores
| model | 80 MHz | 120 MHz | full head 80 / 120 |
|---|---|---|---|
| combo | 108.6 tok/s | **134.2** (+24%) | 85.1 / 108.0 |
| esp32-ai TinyStories | 98.9 | **123.6** (+25%) | 42.4 / 56.1 |
All checks pass at 120 MHz (board == PC bit-exact).

## Decision on the KV DMA pipeline (2026-09-27): not built
The bus test (docs/phase1-bench.md, "PSRAM bus shared") shows ~90 MB/s in total for any mix of cores and GDMA.
So a pipeline cannot make KV bytes cheaper; it can only move them into time when both cores compute from SRAM.
Upper bound on the gain, combo, 2 cores (attention can't be faster than max(hot compute, KV bytes / 90 MB/s)):

| context | attention now | hot compute | KV bytes / 90 MB/s | best case | max gain |
|---|---|---|---|---|---|
| generation, pos ~13-60 (~60 KB) | 1.32 ms | 0.60 | ~0.69 | ~0.69 | ~0.6 ms (7%) |
| pos 128 (~220 KB) | 3.53 | 1.59 | ~2.44 | ~2.44 | ~1.1 ms |

Reaching the bound needs GDMA at the bus's full rate during compute, but GDMA alone gets 68 MB/s, not 90. It also needs ~20-40 KB
of SRAM for buffers taken from weights, and those weights then come from PSRAM: +0.2-0.45 ms per token at 90 MB/s.
That leaves ~0.1-0.3 ms (1-3%) at generation lengths, for a KV layout change and DMA descriptors on both cores.
**Not worth it.** Fewer KV bytes (int4 K/V, ~2x less traffic) would attack the same ~0.7-1.9 ms directly; it needs a quality check first.

## Review fixes (2026-09-29; `engine/out/combo_s0_review.test.log`, `combo_s0_review.board1.log`)
Bugs found by a code review, fixed, then re-checked on the board (combo: 12/12, board == PC bit-exact, 110.0 tok/s as before):

| bug | effect before | fix |
|---|---|---|
| V^T staging block in SRAM was written back only at `pos & 15 == 15` and never reloaded | a step at an earlier position (rewind: speculative rollback, KV reuse) read V of another block. Not hit so far: every caller went forward from 0 | VS tracks which block it holds per (layer, head); entering another block writes the old one back and loads the new one. New test `rewind` (go to 60, redo 37..41): fails on the old code, passes now, PC and board |
| unread head rows got their cluster's int4 centroid score | not a bound on the rows (the centroid is quantized), and equal for a whole cluster: top-k **sampling** (demo, temp 0.7, top-40, head 15%) could pick tokens whose row was never scored | unread rows get `LLM_UNREAD` (-1e30). Tokens, NLL and argmax unchanged; logit digests change (they hash all logits) |
| no range checks: `llm_forward_batch` (pos, token id), `llm_load` (tensor offsets, group length, head offsets, emb_row), demo `G`/`S` commands | a bad pack or command read/wrote out of bounds | checks return NULL / an error code / `ERR ...`; the demo refuses a prompt longer than max_prompt instead of cutting its end (the question) |
| sparse `down` assumed 64-wide groups | wrong result for any ReLU model with another group length | used only when `down` has 64-wide groups |
| `llm_rehead` after a batch used slot 0's input | wrong head for batch > 1 | uses the slot of the last token |

**Speed depends on context length** (new `poscurve` test, combo, 2 cores, 80 MHz, two-stage head, ms/token per 32 positions):

| positions | 13-31 | 32-63 | 64-95 | 96-127 | 128-159 | 160-191 | 192-223 | 224-255 |
|---|---|---|---|---|---|---|---|---|
| ms/token | 8.82 | 9.29 | 10.03 | 10.81 | 11.62 | 12.40 | 13.18 | 13.96 |
| tok/s | 113 | 108 | 100 | 93 | 86 | 81 | 76 | 72 |

The "110 tok/s" headline is positions 13-60. Over a full 256-token context the mean is ~11.3 ms (~88 tok/s):
attention grows ~0.8 ms per 32 positions (KV cache in PSRAM).

**Same-engine comparisons.** The report's "combo is 35% faster than qat4" compared combo on v3 (9.06 ms) with qat4 on
v2 (12.47 ms). On one engine (v2): 10.21 vs 12.47 ms, **22%**.
**PLE on the board** costs 0.9-1.8 ms/token of PLE stages; against its dense trunk (ffn 320 instead of 384) the net
cost is +0.5-0.6 ms/token (ple64 12.99 vs qat4 12.47 ms on v2), not the +0.2 ms the cost model gave for the flash row.
**"Spread < 0.1%"** is repeat-to-repeat spread of one deterministic run (same prompt, same positions); it is not
an uncertainty over prompts or context lengths (see the table above).

## Head partial sums per cluster (2026-09-30; logs `engine/out/*_hpart.test.log`)
The two-stage head kept its int32 group sums in a buffer the size of the whole vocabulary (vout x groups), placed
in SRAM: 32 KB for combo, 64 KB for varC (vocab 8192), ~100 KB for esp32-ai (25,353 head rows). The head only ever
works on one cluster at a time, so each worker now gets a buffer of the largest cluster. Bit-identical (host
digests equal to the previous engine; board 12/12 for all three). The freed SRAM goes to weights:

| model | weights in SRAM before / after | tok/s positions 13-60 (esp32-ai: their bench) | mean over the context |
|---|---|---|---|
| combo | 199 / 223 KB | 110.0 -> **113.0** | ~88 -> ~91 |
| varC | 149 / 203 KB | 94.7 -> **97.9** (cost model: 99) | ~78 -> ~81 |
| esp32-ai | — / 138 KB | 99.5 on their bench (10.05 ms; was 10.87) | ~94 |

esp32-ai's model on our engine is now 9.4x their firmware on their benchmark (94.10 / 10.05 ms).
