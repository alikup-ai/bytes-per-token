# esp32-ai's own model on our engine (2026-09-25)

Model: slvDev/esp32-ai TinyStories release (MIT), 28.9M parameters: 32,768-token input vocabulary,
25,353-row tied head, d 96, 6 layers, 4 heads of 24, ffn 66, PLE 128. Downloaded and checked against
the SHA-256 pinned in their `scripts/fetch_model.sh`. Converted by `engine/export_esp32ai.py`:
**same int4 codes and scales, no requantization** (their integer levels run as `lvl 0` in our
engine; ffn 66 and head dim 24 are zero-padded to 96 and 32, which changes no result).
Reference: their own `src/model.py` with the dequantized weights.

## Correctness (engine/test.py esp32ai_ts --port COM6 --runs 2): 12/12
- Engine NLL 2.1050 vs their PyTorch model 2.1060 on 510 targets. argmax agreement 98.4%.
  Greedy generations identical except one near tie (torch top-2 gap 0.003).
- Two-stage head (256 clusters, 15% = 3,861 of 25,353 rows): top-1 same as the full head 99.8%.
  Read rows are bit-identical to the full head.
- Board == PC bit for bit, PIE 2 cores == PIE 1 core == scalar C, in both runs.

## Speed, ESP32-S3-N16R8, 240 MHz, PSRAM 80 MHz
| engine | head | ms/token | tok/s | vs esp32-ai |
|---|---|---|---|---|
| esp32-ai runtime (their measurement, metadata.json "current", -O3) | full, int8-staged | 94.9 | 9.88 | 1x |
| **ours, 2 cores** | **full** (same computation as theirs) | **24.48** | **40.9** | **3.9x** |
| ours, 1 core | two-stage | 13.87 | 72.1 | 6.8x |
| **ours, 2 cores** | **two-stage, 15% of rows** | **11.37** | **87.9** | **8.9x** |
| ours v3 (per-token work pass, 2026-09-26) | full / two-stage | 23.53 / 10.02 | 42.5 / 99.8 | 4.3x / 10.1x |
| ours v3, PSRAM 120 MHz (experimental, 2026-09-27) | full / two-stage | 17.81 / 8.09 | 56.1 / 123.6 | 5.7x / 12.5x |

Stage split with the full head: head 17.29 ms (1.22 MB streamed from PSRAM), PLE 1.98, FFN 1.83,
attention matvecs 1.72, attention 1.28, input 0.38. With the two-stage head the head drops to 4.18 ms.

The phase-0 cost model predicted 22.3 ms/token for this model with PIE and int4 on both cores
(`costmodel/ideas.py`, measured column). Measured with the full head: 24.48 ms (+10%).
*Corrected 2026-09-29:* 22.3 ms is the prediction at position 128; this measurement ran at positions ~13-60, where
the model gives 21.0 ms, so it is **14% low**. On their benchmark (positions 4-203) 21.95 vs 25.00 ms: 12% low.

## Caveats
- The 94.9 ms is esp32-ai's own published measurement on the same board type, not re-measured
  here. Running their firmware on this board would close that gap.
- Their runtime stages the core and head as int8 in PSRAM (2x the bytes of int4) and uses no SIMD.
  The 3.9x is engine against engine on identical weights. The extra 2.3x of the two-stage head is
  an algorithmic change with a 0.2% top-1 disagreement against the full head.

## Re-measured on our board, same benchmark (2026-09-27)
Their firmware was built from refs/esp32-ai (commit 52ebcbb) with their deploy settings: arduino-cli,
Arduino ESP32 core 3.3.3 (they report 3.3.10), `-O3`, their FQBN and partition table, their model.bin
(boot fingerprint a9bdd778 = their metadata). Two changes outside the timed code: `CDCOnBoot=default`
(console on the UART port our board is wired to) and `USE_DISPLAY 0` (no panel). Their ms/token counts
`llm_forward` only. Benchmark: their prompt "Once upon a time" (ids 433 447 259 405), 200 greedy
tokens, positions 4-203. Capture: `bench/esp32ai_capture.py`, logs `bench/esp32ai_fw_run*.log`.
Our engine ran the same prompt and length (`export_esp32ai.py --their-bench`).

| engine | head | PSRAM | ms/token | tok/s | vs theirs |
|---|---|---|---|---|---|
| esp32-ai firmware (3 runs, identical) | full, int8-staged | 80 MHz | 94.10 | 10.63 | 1x |
| ours | full | 80 MHz | 25.00 | 40.0 | **3.8x** |
| ours | two-stage | 80 MHz | 11.52 | 86.8 | **8.2x** |
| ours | full | 120 MHz (experimental) | 18.90 | 52.9 | 5.0x |
| ours | two-stage | 120 MHz (experimental) | 9.17 | 109.0 | 10.3x |

Their profile on our board: input 2.1, attention 20.5, FFN 6.4, PLE 6.3, **head 58.8** ms/token.
**Same text:** our engine (two-stage head, 80 MHz) generated the same 200 tokens as their firmware:
759 of 759 characters identical.
All our checks pass at both clocks (board == PC bit-exact).

## After the review fixes (2026-09-29, `engine/out/esp32ai_ts_review.test.log`)
Engine with the V^T rewind fix, `LLM_UNREAD` for unread head rows and range checks. 12/12 checks, board == PC
bit-exact, rewind test passes. Their benchmark (prompt "Once upon a time", positions 4-203, 80 MHz):

| head | before | now | vs their 94.10 ms |
|---|---|---|---|
| full | 25.00 ms | 24.81 ms | 3.8x |
| two-stage | 11.52 ms | **10.87 ms** | 8.7x |

Same tokens as before (token digests unchanged). The two-stage gain most likely comes from filling the 21,500
unread logits with one constant instead of a per-token centroid lookup; it was not profiled separately.
The report keeps the earlier 3.8x / 8.2x.

ms/token by context position (two-stage head, 32 positions per bucket from 4): 8.98, 9.68, 10.32, 11.08, 11.76,
12.46, 13.20, 13.90. Like combo, speed falls ~1.6x from the start to the end of the 256-token context.

## Quality against our models, same text (2026-09-30, `train/compare_bpb.py`)
First 5,000 stories of `data/valid.txt` (3.99 MB), each model with its own tokenizer, non-overlapping 256-token
windows, bits per UTF-8 byte. Our QAT models with int4 fake quantization, esp32-ai with its dequantized int4
weights: both as the engine runs them.

| model | stored params | tok/s on our engine (board) | bits/byte | nats/token |
|---|---|---|---|---|
| combo_s0 | 3.4M | 110.0 | 0.732 | 2.008 |
| comboP128_s0 | 4.9M | 103.0 | 0.716 | 1.963 |
| **varC_s0 / s1** | 8.6M | **94.7** | **0.704 / 0.707** | 1.989 / 1.997 |
| varG_s0 | 16.0M | (90 predicted) | 0.706 | 2.003 |
| esp32-ai TinyStories (their release) | 28.9M | 92 on their bench (pos 4-203); ~88 mean over the context | **0.863** | 2.449 |

- Sanity check: esp32-ai's RESULTS.md reports val perplexity 11.41 (2.435 nats/token) in float on their own
  split, and "~ppl +1" after 4-bit. Our 2.449 nats/token for their int4 model on this text agrees, so the text
  does not penalise them noticeably (they trained on the first 300 MiB of TinyStories-train.txt, we on another
  slice; see README.md). No target fell outside their 25,353-row head.
- *Speed at matched positions* (poscurve, our engine, two-stage head, 80 MHz, mean over the 256 context):
  esp32-ai 11.42 ms (~88 tok/s; 111 -> 72), combo 11.3 ms (~88; 113 -> 72), comboP128 ~84, varC 12.75 ms
  (~78; 97 -> 65). The tok/s column above mixes conditions (varC at positions 13-60, esp32-ai on their bench).
- **At equal speed on our engine, combo is 0.13 bits/byte better** than esp32-ai's model with 8.5x fewer
  stored parameters; **varC is ~12% slower and 0.16 bits/byte better** with 3.4x fewer. (Only on our engine:
  their firmware runs their model at 10.6 tok/s and cannot run ours.) Their 28.9M are mostly a 32k-row embedding and PLE table (trunk 558K parameters): on TinyStories a
  32k vocabulary is barely used (973k tokens for this text vs 978k with our 8192 vocabulary), and our variant G
  showed the table saturates. Their weights are also post-training quantized, ours trained for 4 bits.
- So the finding holds on a model we did not train: for this chip, spend the read budget on the trunk and a
  moderate table, not on a huge vocabulary.

## After the head-buffer fix (2026-09-30)
Their benchmark: **10.05 ms/token (99.5 tok/s), 9.4x** their firmware; full head 23.98 ms (3.9x). Mean over the
context ~94 tok/s, combo ~91, varC ~81 (`docs/phase3-engine.md`). So combo is ~4% slower than esp32-ai's model on
our engine and 0.13 bits/byte better; varC trained on 70M tokens (`docs/phase2-h2.md`) reaches 0.625 bits/byte
against their 0.863, ~14% slower. The report keeps the earlier 3.8x / 8.2x for the text race.
