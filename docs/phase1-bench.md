# Phase 1: roofline measured on the board (2026-09-25)

Board: ESP32-S3-N16R8 (chip rev v0.2, 8 MB octal PSRAM @80 MHz, 16 MB QIO flash "boya"),
240 MHz, ESP-IDF 5.5.5, USB-UART CH343 on COM6. Firmware `bench/esp32s3_bench`, captured with
`bench/capture.py`. 5 runs: every number repeated within 0.1% (`bench/roofline_run1..5.json`).
All PIE kernels match the scalar reference bit for bit (`match_ref: 1`).

## Memory
| tier | 1 core | 2 cores | assumed before |
|---|---|---|---|
| SRAM seq read | 295 MB/s | — | "free" |
| PSRAM seq read | 86.7 MB/s | 89.8 MB/s | 72 / 91 (tinyLLM probe) |
| flash seq read (XIP) | 32.0 MB/s | 32.4 MB/s | 13.9 (fitted, Rust engine) |
| flash random row 64 / 256 / 1024 B | 3.8 / 9.7 / 33.2 us | | 20.3 us per 512 B |

## Matrix-vector kernels, 96 columns
| kernel | SRAM cyc/MAC | PSRAM cyc/MAC (weights MB/s) | flash cyc/MAC (MB/s) |
|---|---|---|---|
| scalar int8 | 8.30 | 9.61 (25.0) | 10.2 (23.5) |
| **PIE int8** | **0.386** | 2.98 (80.5) | 7.72 (31.1) |
| scalar int4 | 10.1 | 10.8 (11.1) | 11.1 (10.9) |
| **PIE int4** | **0.469** | 1.64 (73.1) | 4.01 (29.9) |
| PIE int8 / int4, 2 cores, PSRAM | | 86.5 / 83.4 MB/s | |

- PIE is **21x faster than scalar C** at int4 in SRAM (0.47 vs 10.1 cyc/MAC; the scalar reference is a plain
  loop, esp32-ai's tuned C int8 kernel does 6.2 cyc/MAC, so ~13x against real shipping code), twice as fast as
  the 1.0 cyc/MAC the what-if assumed. The esp32-ai claim "SIMD gains ~15%" is wrong by an order of magnitude.
- Streaming from PSRAM, the PIE int4 kernel reaches 73 MB/s on one core and 83 on two:
  84-93% of the bus. Compute hides under memory. **After PIE the chip is memory-bound**
  (PSRAM ~3x slower than compute at int4), not "balanced" as `docs/ideas-costed.md` concluded from the 1.0 assumption.
- Flash streaming is 2.5-3x slower than PSRAM. Keep streamed weights in PSRAM and use flash only for row lookups (PLE, embedding).

## DMA (GDMA async memcpy, PSRAM -> SRAM, 32 KB chunks) while PIE computes on SRAM
| burst | DMA alone | DMA with compute | compute slowdown |
|---|---|---|---|
| 16 B (IDF default) | 32.9 MB/s | 32.9 | +1.3% |
| 32 B | 51.7 | 51.3 | +2.0% |
| 64 B | 68.5 | 67.9 | +2.7% |

- The overlap is real: DMA and compute barely slow each other down.
- *Method caveat, re-measured (review 2026-09-29, `bench/roofline_review.json`, `bench/capture_review.out`):*
  `dma_stream` kept one 32 KB transfer in flight and waited for its interrupt and semaphore before issuing the
  next. With two transfers queued, burst 64: **76.8 MB/s** alone (was 68.6), 76.8 with compute, compute slowed
  3.6%. So "GDMA is slower than the PIE kernel reading PSRAM directly" was partly the method: GDMA now beats
  one core's direct stream (73.1) and stays below two cores' (83.4). The conclusion below still holds, for the
  reason the bus test gives: all agents together get ~90 MB/s, so DMA cannot add bandwidth to the two cores.
  Every other number of the suite repeated the first run (`roofline_run1.json`) within 0.3%.
- But GDMA peaks at 68.5 MB/s, below the 73-83 MB/s the PIE kernel gets by reading PSRAM directly.
  **H4 (DMA double-buffering) does not pay for weight streaming:** the direct kernel already
  gets max(compute, memory). DMA stays useful only to fill SRAM while the core is busy with
  SRAM-resident work (e.g. attention).

## What changes (`costmodel/ideas.py`, "MEAS" column)
Base at 4-bit with PIE and 2 cores: **15.9 ms/token (~63 tok/s) for our 1.8M dense**, 22.3 ms (~45 tok/s)
for esp32-ai's 28.9M TinyStories model. esp32-ai measures 94.7 ms today, so this is 4.2x.
| idea | assumed (overlap) | **measured** |
|---|---|---|
| ReLU gate-first (ours) | x1.22 | x1.22 |
| two-stage head (esp32-ai 25k vocab) | x2.70 | x2.73 |
| 2-bit | x1.51 | x1.52 |
| **speculative, verify 4, accept 2** | x0.93 | **x1.33** (ours), **x1.82** (esp32-ai) |
| SeedLM 4 cyc/w, entropy coding 3 cyc/w | x0.60 | x0.63 (still loses) |

- **Speculative decoding comes back:** with memory-bound decoding, verifying several tokens per
  weight pass pays. The earlier "deprioritize H3" (2026-09-24) is reversed.
- Schemes that pay several cycles per weight (seeds, entropy coding) still lose. The compute budget is
  ~0.5 cyc/MAC, so a decoder must cost well under 1 cycle per weight to help.
- Still assumed: attention on PIE (1 cyc/MAC), per-matvec call cost (10 us), dual-core scaling of attention.

## PSRAM at 120 MHz (2026-09-27; engine/psram120.sh, bench/roofline_120.json)
ESP-IDF marks octal PSRAM at 120 MHz as experimental ("works when the temperature is stable"). On the
ESP32-S3, flash shares the MSPI clock, so flash runs at 120 MHz too. Measured at room temperature, no errors:

| | 80 MHz | 120 MHz | gain |
|---|---|---|---|
| PSRAM seq read, 1 / 2 cores | 86.7 / 89.8 MB/s | 121.8 / 131.2 | +40% / +46% |
| PIE int4 stream from PSRAM, 1 / 2 cores | 73.1 / 83.4 | 100.8 / 121.5 | +38% / +46% |
| flash seq read | 32.0 | 46.5 | +45% |
| flash row 256 B | 9.66 us | 6.62 us | -31% |
| GDMA burst 64 | 68.5 | 68.8 | none (DMA-limited) |

Compute kernels unchanged; all PIE results still bit-exact against the scalar reference.

## PSRAM bus shared by the cores and GDMA (2026-09-27; bench.c section 6, bench/roofline_bus80.json)
80 MHz. Each agent streams its own 1 MB region at the same time; GDMA copies 32 KB chunks to SRAM, burst 64.

| agents | core 0 | core 1 | GDMA | total MB/s |
|---|---|---|---|---|
| core 0 | 83.9 | | | 83.9 |
| cores 0+1 | 44.9 | 44.9 | | 89.8 |
| GDMA, 1 channel | | | 68.2 | 68.2 |
| GDMA, 2 channels | | | 76.8 | 76.8 |
| core 0 + GDMA | 48.8 | | 44.3 | 88.6 |
| cores 0+1 + GDMA | 31.9 | 30.2 | 41.8 | 90.6 |
| cores 0+1 + GDMA x2 | 26.7 | 23.7 | 45.4 | 90.8 |

**The PSRAM bus gives ~90 MB/s in total, whoever reads.** GDMA does not add bandwidth; it only takes a
share of the same ~90 MB/s. It can help only by using the bus while both cores compute from SRAM.
