# Phase 2a: depth vs width at equal bytes/token (2026-09-25)

`train/sweep_depth.sh 0`, `train/sweep_depth.sh 1`. Same setup as H2 (`docs/phase2-h2.md`):
TinyStories, 4096 BPE, 20M tokens, ~901 KB/token at 4-bit, head dim 32, two seeds.
The 6-layer arm is `dense` from H2. Latency is from the cost model, scenario 5 (PIE, int4, both cores; assumed).

| layers | d | ffn | val s0 | val s1 | mean | ms/token (seq / overlap) |
|---|---|---|---|---|---|---|
| 4 | 160 | 384 | 2.0010 | 2.0023 | 2.0016 | **19.1** / 13.6 |
| 6 | 128 | 384 | 2.0000 | 2.0002 | **2.0001** | 20.6 / 14.7 |
| 12 | 96 | 272 | 2.0530 | 2.0858 | 2.0694 | 23.7 / 16.7 |

## Reading
- 4 and 6 layers are tied on quality (+0.0015, inside the seed spread), and 4 layers is
  7% faster on the chip in the model. On quality per ms, 4 layers wins narrowly.
- 12 layers is worse on both axes: +0.07 loss and 15% slower. MobileLLM's
  "deep and thin" result (at ~125M params) does not carry down to ~2M. A width of 96 looks
  too narrow here, and 12 layers also had the largest seed spread (0.033).
- CPU training time follows depth too (12 layers: ~51 min vs ~29-41 min for 4 layers).

## Decision
Keep 6 layers as the reference for comparisons already made. Run new work (phase 2b, larger
PLE) on the 4-layer shape as a second point only if its 7% edge still matters once the board
bench replaces the assumed PIE and call costs: per-call overhead is exactly what 4 vs 6 layers changes.

## Incidents
depth12_s1 died at step 750 (laptop standby, then an unexpected shutdown at 12:29, Kernel-Power 41)
and was retrained from scratch. `train.py` now saves `ckpt.pt` at every eval and resumes
bit-exactly (tested: an interrupted and resumed run matched an uninterrupted one on every logged value).
