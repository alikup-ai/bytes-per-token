# New ways to compute, costed on paper (2026-09-24)

> **Superseded in part.** This page was written with PIE assumed at 1.0 cyc/MAC. The board (phase 1) measured
> 0.47, which makes the chip memory-bound, not "balanced" as the Reading below says. See the MEAS column of
> `costmodel/ideas.py` and `docs/phase1-bench.md`. Speculative decoding later lost again on the tuned engine
> (`docs/phase4-decoding.md`). The table is kept as written.

`python costmodel/ideas.py`. Start point: whatif scenario 5 (PIE 1 cyc/MAC, int4,
int8 KV, both cores, 10 us/call); "ovl" additionally assumes DMA overlap. Knobs
marked [assumed] come from the literature, not from the S3. Quality is not modelled.

## Speed-up vs scenario 5 (seq / with overlap)
| idea | 28.9M TinyStories (esp32-ai) | 1.8M H2 dense (ours) |
|---|---|---|
| 1 skip zero FFN neurons (read 20%) + predictor | 1.02 / 1.01 | 1.29 / 1.32 |
| 1b two-stage head (read 10% of rows) | **2.53 / 2.70** | 1.22 / 1.23 |
| 1+1b | **2.63 / 2.80** | **1.68 / 1.74** |
| 2 weights from PRNG seeds, 2x fewer bytes, 4 cyc/weight | 0.64 / 0.56 | 0.68 / 0.60 |
| 2' same at 1 cyc/weight | 1.10 / 1.14 | 1.08 / 1.11 |
| 3 2-bit + lookup tables (0.5 cyc/MAC) | 1.65 / 1.66 | 1.50 / 1.51 |
| 4 entropy-coded int4, 3 cyc/weight decode | 0.65 / 0.57 | 0.69 / 0.61 |
| 5a speculative, verify 4, accept 2 | 1.01 / 1.09 | 0.93 / 0.93 |
| 5b 30% of tokens forced by a JSON schema | 1.43 / 1.43 | 1.43 / 1.43 |

## Reading
- After PIE the S3 is *balanced*: memory is only ~2x compute (19.5 vs 9.4 ms).
  GPU-style "spend idle compute to read fewer bytes" (seeds, entropy coding,
  speculative verification) does not pay here: there is little idle compute.
- What pays is **skipping work entirely**: fewer bytes and fewer MACs at once.
  Where to skip depends on where the bytes are: the head for a big vocabulary,
  the FFN for a small one. Measure the per-stage byte split before choosing.
- Schema-forced tokens (5b) are free speed for tool-call output, no quality risk.

## Prior work (searched 2026-09-24): all exist on GPU/CPU/FPGA, none found on an MCU
- 1: Deja Vu, PowerInfer, SparseInfer (arXiv 2411.12692), ReLU^2 (2402.03804), SLIM (2507.09201).
- 1b: Learning to Screen, L2S (arXiv 1810.12406): ~10x faster softmax, targeted at mobile.
- 2: SeedLM (Apple, ICLR 2025, arXiv 2410.10714), FPGA.
- 3: T-MAC (Microsoft, EuroSys 2025, arXiv 2407.00088), CPUs down to Raspberry Pi 5.
- 5a: n-gram drafts from the task corpus (TASC, arXiv 2602.24174); mobile NPU work.
- 5b: jump-forward decoding (SGLang, mlc-llm PR 2551, XGrammar).
"Not found on MCU" = not in these searches, not proof of absence.

## Candidates for training runs
1. Two-stage head + FFN sparsity (1+1b): biggest gain, needs quality check (recall of the screen, ReLU-fied FFN).
2. 2-bit + lookup tables (3): needs quality at 2 bits and a PIE LUT kernel on the board.
3. Jump-forward on schema tokens (5b): engineering only, do it in the engine.
