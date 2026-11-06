# Phase 2, H2 first pass: where to put extra parameters at equal bytes/token (2026-09-24)

`train/sweep_h2.sh 0`, then `python train/report_h2.py`.
TinyStories, shared 4096-token BPE, 20M training tokens per arm, same token stream
(seed 0), 6 layers, d=128. Final val loss on 1M held-out tokens. Latency is from
the phase-0 cost model at position 128: "now" = esp32-ai C engine as measured,
"opt" = whatif scenario 5 (PIE, int4, int8 KV, both cores; assumed).

| arm | stored M | KB/token @4b | val loss | ppl | ms/tok now | ms/tok opt |
|---|---|---|---|---|---|---|
| dense_f320 (ple64's trunk, no table) | 1.65 | 827 | 2.0059 | 7.43 | 65.3 | 19.1 |
| dense | 1.80 | 901 | 2.0000 | 7.39 | 69.2 | 20.6 |
| moe4 (top-1 of 4 experts) | 4.46 | 903 | 1.9905 | 7.32 | 69.8 | 20.7 |
| **ple64** (per-layer table, 1 row/token) | 3.38 | 901 | **1.9607** | **7.10** | 71.1 | 20.8 |

## Seed 1 (2026-09-25; seed changes init and the training token order)
| arm | val s0 | val s1 | mean | vs dense (s0, s1) | vs dense_f320 (s0, s1) |
|---|---|---|---|---|---|
| dense_f320 | 2.0059 | 2.0116 | 2.0087 | +0.006, +0.011 | — |
| dense | 2.0000 | 2.0002 | 2.0001 | — | — |
| moe4 | 1.9905 | 1.9912 | 1.9909 | -0.010, -0.009 | — |
| **ple64** | 1.9607 | 1.9783 | **1.9695** | -0.039, -0.022 | **-0.045, -0.033** |

Seed-to-seed spread per arm: 0.0002-0.018. Every paired gap keeps its sign on both seeds.
**H2 holds on this setup:** PLE beats its own trunk by 0.033-0.045 and dense by 0.022-0.039
at the same bytes read per token. MoE's gain is consistent but small (0.01).
PLE also has the largest spread (0.018), so its size of gain is known only to about ±0.01.
Two seeds cannot give a proper confidence interval; a third seed on dense/ple64 would.

## Reading of seed 0 alone (written before seed 1)
- **PLE helps at equal bytes read:** -0.045 against its own trunk (dense_f320),
  -0.039 against dense. The table is 1.57M params (~0.8 MB flash at 4-bit),
  read at ~200 B/token. Cost: +0.2 ms/token (flash row reads) in the model.
- **MoE top-1 barely helps:** -0.0095 against dense, with 2.5x the stored parameters.
  At this size each expert sees a quarter of the tokens; the router adds nothing
  the trunk lacked.
- Trimming the FFN 384 -> 320 cost only 0.006: the FFN is not where this model is short.
- The same-step curves agree: PLE led from step 250 on (e.g. 2.357 vs 2.407 dense at step 750).

## Scope of the result (review note, 2026-09-29)
Every arm trained on the same 20M tokens with one learning rate (3e-3) and schedule, tuned for none of them.
That is ~11 tokens per stored parameter for dense, ~6 for PLE (3.38M) and ~4.5 for MoE (4.46M): the arms with
more stored parameters are further from convergence. So H2 is shown **at this token budget**; whether PLE's lead
grows or shrinks with more data is not tested. The same holds for phase 2a (one LR for d = 96..160).

## A larger table at the same bytes read (2026-09-29, `train/sweep_ple128.sh`)
combo (PLE + ReLU + QAT 4-bit) with the per-layer table twice as wide (P 64 -> 128) and FFN 320 -> 256, so the
parameters read per token stay equal (1.803M); stored parameters grow 3.38M -> 4.95M (+0.9 MB of flash).

| arm | P | ffn | s0 | s1 | mean | vs combo (s0, s1) |
|---|---|---|---|---|---|---|
| combo | 64 | 320 | 2.0082 | 2.0116 | 2.0099 | — |
| **comboP128** | 128 | 256 | **1.9634** | **1.9778** | **1.9706** | **-0.045, -0.034** |

Better on both seeds by 7-10x combo's seed spread: at equal bytes read, moving parameters from the FFN into the
flash table pays, and the gain (-0.039) is larger than PLE's first gain over dense (-0.022..-0.039, fp32) even
though these are 4-bit QAT arms. The table's growth has not saturated at P=128.

On the board (`engine/test.py comboP128_s0 --port COM6`, log `engine/out/comboP128_s0.test.log`): 12/12 checks,
board == PC bit-exact, **103.0 tok/s** (9.71 ms) at positions 13-60 against combo's 110.0; the calibrated cost
model predicted 104. Over the context: 106 -> 69 tok/s (9.46 -> 14.56 ms per 32-position bucket), ~84 tok/s on
average. Full head 81.1 tok/s. So -0.039 val loss costs ~6% speed and 0.9 MB of flash.
The larger-vocabulary variant (8192 tokens, P=128, 8.6M stored, predicted ~99 tok/s) is the next candidate.

## Variants from the calibrated cost model (2026-09-29/30, `train/sweep_variants.sh`)
All combo-style (ReLU gate, QAT 4-bit, PLE, two-stage head), 20M training tokens, lr 3e-3. Vocabularies differ,
so quality is **bits per byte** of validation text (`val_bpb` in final.json, `train/data.py bits_per_byte`),
not loss per token. Bytes per token: 4.03 (4096), 4.16 (8192), 4.17 (16384): on TinyStories a larger
vocabulary barely shortens the text, so what it adds is PLE table rows.

| arm | vocab | P | d x layers | stored | bpb s0 | bpb s1 | mean | vs combo | tok/s board (predicted) |
|---|---|---|---|---|---|---|---|---|---|
| combo | 4096 | 64 | 128 x 6 | 3.4M | 0.7325 | 0.7337 | 0.733 | — | 110.0 |
| A comboP128 | 4096 | 128 | 128 x 6 | 5.0M | 0.7162 | 0.7214 | 0.719 | -0.014 | 103.0 (104) |
| B varB | 8192 | 64 | 128 x 6 | 5.5M | 0.7206 | — | 0.721 | -0.012 | (104) |
| D varD | 4096 | 128 | 160 x 4 | 4.2M | 0.7184 | 0.7091 | 0.714 | -0.019 | (102) |
| **C varC** | 8192 | 128 | 128 x 6 | 8.6M | **0.7049** | **0.7075** | **0.706** | **-0.027** | **94.7** (99) |
| G varG | 16384 | 128 | 128 x 6 | 16.0M | 0.7058 | — | 0.706 | -0.027 | (90) |

- **C is the best trade.** Seeds agree (0.705 / 0.708). A wider table (A) and more table rows (B) add up:
  -0.014 + -0.012 ~ -0.027.
- **The table saturates here:** G stores twice C's parameters and is no better. Rare tokens of a 16k vocabulary
  are seen too seldom in 20M tokens; more data, not more rows, is the next lever.
- D (wider, shallower trunk) helps a little but its seeds disagree (0.718 / 0.709).
- **C on the board** (`engine/out/varC_s0.test.log`): 12/12, board == PC bit-exact, 94.7 tok/s at positions
  13-60, 97 -> 65 tok/s over the context (~78 mean), full head 61.7. 4.5% below the prediction: the larger
  vocabulary's head buffers (partial sums, logits) take SRAM, so fewer weights fit there (149 KB vs 194 KB);
  the cost model does not price SRAM placement.

## Not yet known
- Seed noise. A 0.006 gap (dense vs dense_f320) may be noise; 0.04 probably is not.
  Seed 1 (`logs/train/runs_h2_s1.log`) decides.
- Whether PLE's gain holds on the tool-call task (b) and at other depth/width shapes (phase 2a).
- Whether a larger table (P=128) or putting the saved bytes into the trunk does better.

## Housekeeping
Run 1 of dense_s0 died at step 1750 (session closed / laptop on battery). The sweep
now skips finished arms and is launched detached; the restart reproduced step-250
loss exactly (deterministic). Empty bench dirs runs/b1..b3 removed.

## Data is the next lever (2026-09-30, `train/sweep_long.sh`)
varC trained on 70M tokens (~0.9 epoch of `train_v8192.bin`) instead of 20M, same shape, same schedule shape:

| arm | training tokens | val bpb | bpb on the shared 5,000-story text (`compare_bpb.py`) |
|---|---|---|---|
| varC_s0 | 20M | 0.7049 | 0.7040 |
| **varC_long_s0** | **70M** | **0.6261** | **0.6253** |

-0.079 bits/byte at unchanged speed on the board: 3x what all the shape changes gave (combo -> C, -0.027).
With the table saturated (varG), more data is where quality comes from. One seed; ~2.6 h on the laptop CPU.
Speed of this shape after the head-buffer fix (`docs/phase3-engine.md`): 97.9 tok/s at positions 13-60, ~81 mean.
