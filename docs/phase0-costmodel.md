# Phase 0: cost model (2026-09-24)

`python costmodel/validate.py` then `python costmodel/whatif.py`.

## Model
Per op: `compute + bytes / bandwidth` (additive: in-order LX7, blocking misses;
esp32-tinyLLM's int8-head experiment shows the extra bytes cost their full
bandwidth price). Bytes, MACs and call counts come from shapes and each engine's
placement rules (`engines.json`, read from source). Five per-engine cycle costs
are fitted by NNLS on calibration runs only.

## Validation (held-out, never used in fitting)
| test | measured | predicted | error |
|---|---|---|---|
| Barista 8.9M (different D, F, head) from TinyStories-fitted esp32-ai engine | 49.6 ms/forward | 51.1 | +3.1% (mean_pos assumed 24; 8..48 gives -2%..+11%) |
| Rust engine, PSRAM 120 MHz, from 80 MHz fit | 91.7 ms/token | 87.5 | -4.6%, worst stage 7.3% (head) |

Calibration fits are near-exact by construction (4-5 parameters, 5 stages) and
say nothing on their own. The held-out rows are the evidence.

Fitted values: esp32-ai C int8 kernel 6.2 cycles/MAC (matches tinyLLM's own
estimate of ~6 for C), 80 us per matvec call; Rust int4 kernel 7.1 cycles/MAC,
26 us per call, flash XIP effective 13.9 MB/s.

Known weaknesses: head under-predicted at 120 MHz (the head partially hides
memory behind its issue-bound loop, which the additive form ignores);
transcendental cost fitted to 0 for Rust (absorbed elsewhere, not identifiable
from 5 stages); only two engines and one board.

## Review notes (2026-09-29)
- The validation is thin: two held-out points. Barista depends on an assumed mean position (−2%..+11% over 8..48),
  and the fast-MSPI run changes only the PSRAM clock. `measured.json` says the flash "stays at its own clock"
  there, but phase 1 found that on the S3 flash shares the MSPI clock; if tinyLLM's flash ran faster too, the
  −4.6% agreement partly comes from errors cancelling.
- `hw/esp32s3.json` still holds the pre-board figures (PSRAM 72/91 MB/s, 20.3 us per flash row). The board values
  (phase 1) are applied only in `ideas.py` ("MEAS" column); the calibration fits keep the old ones on purpose,
  because those engines were measured by their authors.
- Scale bytes: `arch.py` charged one scale per row. It now takes a `group` per tensor (our engine: fp32 per 64
  weights, 6% of bytes). The calibration engines are unchanged (one group per row).
- Checked against our own engine at matched positions, the model is 10-14% low (`ideas.py`, last table), not
  "+4%/+10%" as phase 3 first wrote: those compared a position-128 prediction with runs at positions 13-60.

## Correction to an earlier claim
esp32-ai's RESULTS.md calls the head "PSRAM-bandwidth-bound" and bounds SIMD at
~15%. The model and tinyLLM's probes disagree: in the shipping C engine 53 of
95 ms are compute (6.2 cycles/MAC scalar). tinyLLM measured its head at 21% of
bus bandwidth, 8.13 cycles/MAC cache-resident vs 9.01 streaming (issue-bound).

## What-if (esp32-ai engine, TinyStories, mean pos 103.5)
| scenario | ms/token | tok/s | compute | memory |
|---|---|---|---|---|
| 0 baseline | 94.7 | 10.6 | 53.0 | 41.7 |
| 1 PIE matvec 1.0 cyc/MAC [assumed] | 59.6 | 16.8 | 17.9 | 41.7 |
| 2 + everything on 2 cores | 53.7 | 18.6 | 14.1 | 39.6 |
| 3 + int4 weights in PSRAM | 37.3 | 26.8 | 14.1 | 23.1 |
| 4 + int8 KV, PIE attention | 31.4 | 31.9 | 12.2 | 19.2 |
| 5 + call overhead 80 -> 10 us | 28.3 | 35.4 | 9.1 | 19.2 |
| 6 + DMA overlap (max, not sum) | 20.3 | 49.3 | 9.1 | 19.2 |
| 7 = 6 with 2-bit weights | 12.1 | 82.6 | 9.1 | 10.9 |

Reading: today the chip is compute-bound (scalar code). Once SIMD lands
(1-2), memory dominates and every further step is a bytes-per-token step
(3, 4, 7) or an overlap step (6). That is H1 in the plan, now quantified, and
it is why phases 2 and 4 (architecture and multi-token decoding at fixed bytes
per token) are the research core.

Assumptions to replace with phase 1 measurements: PIE cycles/MAC including
int4/int2 unpack, whether GDMA can stream PSRAM->SRAM concurrently with compute,
dual-core bus scaling for mixed workloads. Scenario 7 also assumes 2-bit weights
keep quality, which only phase 2 can answer.
