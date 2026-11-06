# Phase 4: several tokens per weight pass — first look (2026-09-26)

## Drafter acceptance (decode/spec_sim.py, combo_s0, 40 val prompts x 64 greedy tokens)
Tokens yielded per weight pass when D drafted tokens are verified at once (1 + accepted prefix):

| drafter (no extra model) | D=1 | 2 | 3 | 4 | 8 |
|---|---|---|---|---|---|
| context n-gram (last 3/2/1 tokens seen before) | 1.15 | 1.20 | 1.22 | 1.22 | 1.23 |
| bigram table (most frequent next token, 8 KB) | 1.35 | 1.48 | 1.52 | 1.53 | 1.53 |
| context, else bigram | 1.39 | 1.54 | 1.60 | 1.62 | 1.64 |

## Why it does not pay on engine v2 (estimate from the measured stage split)
Combo on the board: 10.22 ms/token = **W 6.87 ms** of weight streaming, which a batch of positions
would share, + **C 3.35 ms** of per-position work (attention, int8 quantization, norms, GELU,
head selection), which a batch repeats for every verified position. A pass verifying D drafts costs W + (D+1)C:

| D | pass | tokens | speed-up |
|---|---|---|---|
| 1 | 13.6 ms | 1.39 | x1.05 |
| 2 | 16.9 ms | 1.54 | x0.93 |
| 3 | 20.3 ms | 1.60 | x0.81 |

The phase-0 cost model predicted x1.33 because it treated decoding as almost purely memory-bound.
Engine v2 removed most of the memory-bound work (int4, SRAM, two-stage head, sparse FFN), so per-token
compute is now a third of the time, and verifying extra positions costs almost as much as it saves.
**Finding:** on this chip, speculative decoding pays only when W >> C. This holds for esp32-ai's model
with the full head (W ~22 ms, C ~2.7 ms: estimated x1.25), but not for our tuned models.

*Review note (2026-09-29): the estimate W + (D+1)C is optimistic, so the negative result is, if anything, stronger.*
(1) Verification needs logits at every one of the D+1 positions, but `llm_forward_batch` computes the head only for
the last token; with the two-stage head each position selects its own clusters, so the head's weight reads are not
shared either. (2) The batch table in `docs/phase3-engine.md` shows W is only partly shared from the cache (2 tokens: 13.96 ms per pass
measured vs ~11.8 by the formula). (3) Acceptance was measured on greedy output only; the demo samples
(temperature 0.7, top-40), where drafts are accepted less often. A rejected draft also needs a rewind of the KV
cache: that path had a bug in the V^T staging (fixed 2026-09-29, `docs/phase3-engine.md`).

## What would change the answer
1. Cut C: attention (1.5 ms) and element-wise work are now the bottleneck, and each ms saved speeds plain
   decoding too.
2. A better drafter (a tiny draft model): even E = 2.5 at D = 4 gives only x1.05 at the current C.
3. Forced tokens need no drafter (prompt prefill, schema-fixed JSON spans: E = T). A batched pass saves
   (T-1) x W, e.g. 5 forced tokens: 51 ms sequential vs ~24 ms batched.

## After engine v3 (per-token work cut, 2026-09-26)
Combo: 9.14 ms = W ~6.5 + C ~2.65. Speculative with the context+bigram drafter: D=1 x1.08, D=2 x0.97.
Still marginal: the drafter (1.4-1.6 tokens/pass) is now the limit, not C alone.

## Template filling: tokens fixed by the template are batched (2026-09-27, plan item 3)
Demo firmware command `S` (engine/esp32/main/demo.c): a template is fixed text and slots. Fixed text goes through
`llm_forward_batch` (batches of up to 16); a slot is generated token by token until the model wants a token that
starts with punctuation (19 such tokens in the 4096 vocabulary, 115 in esp32-ai's), which the template then supplies.
`engine/template_test.py` runs each template with batching and token by token (baseline), greedy, best of 3.
Logs: `engine/out/template_combo80.log`, `template_esp32ai80.log`. Board, 2 cores, PSRAM 80 MHz.

| model | template | generated + fixed | token by token | batched | gain | same tokens |
|---|---|---|---|---|---|---|
| combo | "Once upon a time ... ___ ..." | 30 + 33 | 574.4 ms | 468.1 | x1.23 | yes |
| combo | "Tom had a big red ___ ..." | 29 + 30 | 533.8 | 440.5 | x1.21 | yes |
| esp32-ai | "Once upon a time ... ___ ..." | 34 + 33 | 672.6 | 494.5 | x1.36 | yes |
| esp32-ai | "Tom had a big red ___ ..." | 19 + 30 | 483.9 | 327.9 | x1.48 | yes |

- Batched output is bit-identical to token by token in every run: the gain costs no quality.
- Fixed text alone gets x1.56 (combo) to x2.2 (esp32-ai). The overall gain is smaller because these fixed spans
  are short (3-8 tokens, where batching gives x1.5-2 per the batch table) and the slots still run at normal speed.
- Rule of thumb: overall gain = 1 / (g + f/k) with g, f the generated and fixed shares of the tokens and k the
  batch gain on fixed text. A JSON answer with ~60-70% fixed tokens in longer spans should give x1.5-2 on esp32-ai.
