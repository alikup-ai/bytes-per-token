# Phase 2b: skip work, fewer bits (started 2026-09-25)

## Part 1: no training (`train/eval_2b.py`, 200k val tokens, seed 0 arms)

### Post-training quantization (group 64, fp16 scale)
| arm | fp32 | 4-bit | 2-bit |
|---|---|---|---|
| dense | 1.9862 | 2.0221 (+0.036) | 3.4375 (+1.45) |
| ple64 | 1.9462 | 1.9786 (+0.032) | 3.1774 (+1.23) |
| depth4 | 1.9859 | 2.0234 (+0.038) | 3.3763 (+1.39) |

- All H2/2a losses so far were fp32. At the 4-bit the cost model assumes, each arm loses
  ~0.035, the same size as PLE's gain. The PLE gain survives quantization (-0.044 at 4-bit).
- 2-bit post-training is unusable. QAT arms (`qat4`, `qat2`) are training now.
- *Review 2026-09-29:* this PTQ is the naive baseline (absmax scale, no search; the scale was kept in fp32, not
  fp16 as written above). With the clipping ratio chosen per group by weight MSE (`train/eval_2b.py --clip`,
  log `logs/train/runs_ptq_clip.log`) the same half-integer grid gives:

  | arm | fp32 | 4-bit absmax | **4-bit clipped** | 2-bit absmax | **2-bit clipped** |
  |---|---|---|---|---|---|
  | dense | 1.9862 | +0.036 | **+0.031** | +1.45 | **+0.63** |
  | ple64 | 1.9462 | +0.032 | **+0.029** | +1.23 | **+0.69** |

  A symmetric integer grid with a zero level (one level fewer) is worse at both widths. So 4-bit PTQ is a little
  better than first reported, and QAT's edge over PTQ is ~0.007 smaller; 2-bit PTQ is poor rather than
  "unusable", still twice the loss of 2-bit QAT (+0.30).

### Two-stage head, post-hoc k-means (64 clusters on the tied embedding)
| rows read | recall of true token | top-1 same as full head | loss with centroid stand-in |
|---|---|---|---|
| 5.9% | 0.89 | 0.92 | +0.60 |
| 11% | 0.945 | 0.97 | +0.30 |
| 21% | 0.975 | 0.99 | +0.14 |
(dense; ple64 and depth4 within ±0.01)

- For greedy decoding, reading 11% of the head rows picks the same token 97% of the time,
  with no training at all.
- The loss column overstates the damage: unread rows get their centroid's logit, a crude
  stand-in. A trained screen (L2S) and balanced clusters (sizes here run 1..271) should do better.
- On our 4096-token vocabulary the head is only ~29% of bytes, so the cost model gives ×1.2 here.
  The ×2.5 case is a large-vocabulary model like esp32-ai's 25k head. Screening matters when the vocabulary is large.

## Part 2: training (`train/sweep_2b.sh 0` and `1`, running)
relu, relu2 (sparse FFN gate) and qat4, qat2, each on two seeds.

### Sparse FFN, seed 0 (final val loss, 1M tokens; dense SwiGLU = 2.0000)
| arm | val loss | live FFN neurons per layer (L1..L6) | mean live |
|---|---|---|---|
| relu | 1.9986 | 42.9 13.3 15.9 16.7 20.3 24.8 % | **22.3%** |
| relu2 | 2.0228 | 50.3 20.6 24.6 26.4 31.1 37.1 % | 31.7% |

- ReLU costs nothing in quality (-0.001) and leaves ~22% of neurons live: an oracle
  could skip ~78% of FFN reads, close to the 20% the cost model assumed (x1.29 on this shape).
- ReLU^2 is worse on both counts here. Layer 1 is the least sparse in both.
- Next: a low-rank predictor per layer (the oracle is not available at run time); measure its recall and the loss when only predicted neurons run.

### Skipping dead neurons at run time (`train/predictor_2b.py relu_s0`, 200k val tokens, base 1.9831)
| method | FFN bytes read | val loss change |
|---|---|---|
| gate-first (compute gate, read up/down only where live), exact | 48.2% | 0 |
| predictor r=32, thr 0.5 | 39.1% | +0.0077 |
| predictor thr 0.3 | 44.9% | +0.0029 |
| predictor thr 0.2 | 48.8% | +0.0015 |
| predictor thr 0.1 | 55.1% | +0.0005 |

- The predictor (11% of FFN bytes by itself, recall 0.92-0.99) barely beats the exact
  gate-first method, and only by giving up quality. **Use gate-first:** no predictor, no loss,
  simpler engine (store `down` transposed so a live neuron is one contiguous row in each matrix).
- Cost model with gate-first (48% of FFN read): x1.21 on our shape, x1.02 on esp32-ai's
  (its FFN is tiny, 66 wide). Combined with the two-stage head this reaches the x1.7 "1+1b" row.

## Part 2 results, two seeds (final val loss, 1M tokens; dense = 2.0000 / 2.0002)
| arm | s0 | s1 | mean | vs dense | live FFN neurons (s0, s1) |
|---|---|---|---|---|---|
| **relu** | 1.9986 | 1.9968 | 1.9977 | **-0.002** | 22.3%, 20.9% |
| relu2 | 2.0228 | 2.0034 | 2.0131 | +0.013 | 31.7%, 32.2% |
| **qat4** | 2.0201 | 2.0284 | 2.0242 | **+0.024** | — |
| qat2 | 2.3042 | 2.3053 | 2.3047 | +0.305 | — |

- **ReLU is free:** same quality as SwiGLU on both seeds, ~79% of FFN neurons dead, exact
  gate-first skipping reads 47-48% of FFN bytes (x1.2 on our shape in the cost model). Adopt ReLU.
- **4 bits is nearly free with QAT:** +0.024 (post-training rounding costs +0.036).
- **2 bits at equal size costs +0.30.** QAT rescues it from collapse (+1.45 post-training), but the
  fair test is 2 bits with 2x the parameters at the same bytes: `train/sweep_2bw.sh` is ready,
  and was held back at the user's request (2026-09-25).
- Not yet measured: PLE + ReLU + QAT4 combined, and whether their gains add up.

## 2 bits at equal bytes (`train/sweep_2bw.sh 0` and `1`, 2026-09-27)
qat2w: d 192, 6 heads, ffn 560, 3.61M weights at 2 bits, against qat4: d 128, 1.80M weights at 4 bits.
With the fp32 scale per 64 weights, qat2w reads 2.5 bits x 3.61M = 1.13 MB/token, qat4 4.5 x 1.80M = 1.01 MB (qat2w reads 11% more).

| arm | s0 | s1 | mean | vs qat4 |
|---|---|---|---|---|
| qat4 | 2.0201 | 2.0284 | 2.0242 | — |
| qat2 (same width) | 2.3042 | 2.3053 | 2.3047 | +0.281 |
| **qat2w (2x weights)** | 2.2459 | 2.1422 | 2.1941 | **+0.170** |

- **Negative:** doubling the width wins back only 0.11 of the 0.28 lost to 2 bits. Both seeds are worse than
  qat4 (+0.23, +0.11), while reading 11% more bytes. qat2w has a large seed spread (0.10). At this model size and
  20M training tokens, 4 bits is the best point, and cost-model scenario 7 (2 bits, x1.5) is not reachable without losing quality.
- No 2-bit kernel was written for the engine: a quality loss this large makes the speed question moot.
- *Review caveats (2026-09-29):* (1) qat2w has 2x the parameters on the same 20M tokens (~5.5 tokens per
  parameter), so it is further from convergence than qat4; (2) lr 3e-3 was not re-tuned for d=192, and the 0.10
  seed spread points to unstable training; (3) `FakeQuant` cut its groups from the flattened tensor, so in `down`
  (560 columns, not a multiple of 64) groups crossed row boundaries, a layout the engine cannot run. FakeQuant now
  groups along rows (identical for every other arm, checked bit for bit). The conclusion is therefore "2 bits lose
  at this token budget and LR", not in general. A rerun with lr 1.5e-3 and row groups (`qat2w_lr15_s0`,
  `train/sweep_review.sh`) is reported below.

### Rerun with a lower learning rate (2026-09-29)
| arm | lr | val loss | vs qat4 mean (2.0242) |
|---|---|---|---|
| qat2w_s0 | 3e-3 | 2.2459 | +0.222 |
| qat2w_s1 | 3e-3 | 2.1422 | +0.118 |
| **qat2w_lr15_s0** (row groups) | 1.5e-3 | **2.1580** | **+0.134** |

At the same steps the new run tracks the good seed (s1) and beats the bad one (s0) by ~0.1 from step 750 on,
so the 0.10 spread was mostly unstable training at lr 3e-3. **The negative result stands:** 2 bits at equal
bytes lose ~0.12-0.13 to 4 bits, a smaller and steadier gap than the +0.17 averaged over the unstable seeds.
Still one seed at the new LR and the same 20M-token budget.
