# Engine v1: design (2026-09-25)

Goal: run our trained models (dense / ReLU / PLE / QAT4) on the ESP32-S3 and on the PC
with **bit-identical** results, then measure real ms/token against the cost model.

## What the numbers say to optimise (docs/phase1-bench.md)
- Weights streamed from PSRAM: 73 MB/s on one core, 83 on two. PIE int4 compute is 0.47 cyc/MAC,
  about 3x faster than PSRAM can feed it. **So bytes read per token decide speed**; compute is
  cheap enough to spend on anything that saves bytes.
- SRAM streams at 295 MB/s and PIE runs at full speed from it: every weight byte moved into
  SRAM is read ~3.5x faster.
- Flash streams at 32 MB/s: never stream weights from it. Row lookups (embedding, PLE table) are fine (~10 us / 256 B).
- DMA double-buffering is slower than the direct PIE stream (68.5 < 83 MB/s): not used.

## Decisions
| # | decision | why |
|---|---|---|
| 1 | int4 weights, group 64, fp32 scale (fp16 tried in v2: slower), **same rounding as training** (`FakeQuant`) | QAT models run exactly as trained; group 64 = 2 PIE blocks |
| 2 | int8 activations, one symmetric scale per vector | PIE `vmulas.s8` needs int8 on both sides; error measured in tests |
| 3 | PIE kernel returns one int32 per 64-weight group; C folds scales | per-group scales + the +0.5 level offset: `dot = s*sa*(2P + A_g)/2`, exact integer part |
| 4 | **both cores** split every matvec by rows; core 1 spins on a job flag | +14% bandwidth (83 vs 73 MB/s); spinning avoids ~10 us FreeRTOS wake-up per call |
| 5 | **SRAM** filled greedily with whole tensors after the engine's own buffers, 32 KB kept free (v2) | those bytes stream 3.5x faster; the rest stays in PSRAM |
| 6 | fused calls: q,k,v one matrix; gate and up issued together | fewer dispatches (per-call cost is still unmeasured) |
| 7 | **ReLU gate-first**: compute gate, run `up` only on live rows; `down` from transposed codes of live rows only (v2, PIE QACC kernel) | exact: the same integer sums as the dense path |
| 8 | int8 KV cache in PSRAM; K as [head][t][32], V transposed [head][32][t] | both attention products become PIE s8 gemv: scores = K q, out = V^T p |
| 9 | PLE table and embedding stay in **flash** (mmap), one row per token | 0.8 MB table read at ~200 B/token |
| 10 | RoPE tables exported from PyTorch, not recomputed | no sin/cos on the chip, identical values |

| 11 | **two-stage head** (v2): 256 k-means clusters of output rows, centroid scores, read the best clusters up to 15% of rows | read rows bit-identical to the full head; top-1 agreement 99.6% |
| 12 | unread head rows get `LLM_UNREAD` (-1e30), not their centroid score (2026-09-29) | argmax and top-k sampling can only pick scored tokens |
| 13 | V^T staged per 16-position block in SRAM; a step into another block writes the staged one back and loads the new one (2026-09-29) | positions may go back (rewind), e.g. speculative rollback; test `rewind` |
| 14 | `llm_load` checks every tensor offset against the pack; `llm_forward*` return NULL for a position outside [0, ctx) or a token outside the vocabulary | a bad pack or host command cannot read or write out of bounds |

## Bit-exactness (PC == board, scalar == PIE, 1 core == 2 cores)
- Integer kernels are exact by construction; every float op runs in the same order in the same C code.
- No `libm` in the forward pass: own `exp`, `log`, reciprocal and rsqrt from IEEE `+ - *` only
  (newlib and the PC libm differ in the last bit; Xtensa has no hardware divide/sqrt).
- `-ffp-contract=off` on both compilers: Xtensa GCC otherwise fuses `a*b+c` into `madd.s`.
- Each row's float epilogue is computed by whichever core owns that row, with the same code, so the split does not change results.

## Tests (engine/test.py)
1. PC engine vs PyTorch (teacher-forced on val text): NLL difference, top-1 agreement.
2. PC twice: identical digest.
3. Board: PIE vs scalar kernels, 1 vs 2 cores: identical digests.
4. Board vs PC: identical generation digest and identical NLL bits.
5. Speed: ms/token by stage, 5 repeats, against `costmodel` predictions.

## Next
Speculative / multi-token decoding (plan phase 4). Per-row scales need QAT at row granularity
(fp16 group scales were slower, see docs/phase3-engine.md). Attention for long contexts.
