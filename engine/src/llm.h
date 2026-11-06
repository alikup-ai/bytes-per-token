// Portable decoder for the mcu-lm-research models (engine/DESIGN.md).
// Same C on the PC and the ESP32-S3; only the kernels and the two-core
// dispatcher come from the platform (struct llm_platform).
#pragma once
#include <stddef.h>
#include <stdint.h>

#define LLM_MAXL 16
#define LLM_MAXB 16      // max tokens per llm_forward_batch
#define LLM_MAXG 32      // max 64-wide groups in the FFN (ffn <= 2048)

enum { ACT_SILU, ACT_RELU, ACT_RELU2 };
enum { ST_INPUT, ST_ATTN_MV, ST_ATTN, ST_FFN, ST_PLE, ST_HEAD, ST_N };

// int4 matrix in PIE blocks (32 codes per 16 bytes). A row is `groups` scale
// groups of `bpg` blocks each (our models: 64 codes = 2 blocks; esp32-ai: one
// group per row). lvl 1: w = (code - 7.5) * s (our FakeQuant); lvl 0: w = (code - 8) * s.
// s: fp32 scale per (row, group); s16: fp16 instead (row lookups only, no matvec).
typedef struct { const uint8_t *w; const float *s; const uint16_t *s16; int rows, cols, groups, bpg, lvl; } qmat_t;

typedef struct {
    int vocab, d, layers, heads, ffn, ple, act, ctx;
    int vout;                   // logits per step (<= vocab); the head is the first vout embedding rows
    int lvl;                    // quantizer levels of every tensor (see qmat_t)
    const uint32_t *emb_row;    // optional: token -> embedding row (single-copy clustered head)
    qmat_t emb, ple_table, ple_mproj;
    // two-stage head (optional): rows in cluster order, centroids, row -> token, cluster offsets
    qmat_t head, head_cent;
    const uint32_t *head_perm, *head_off;
    int clusters;
    const float *norm, *ple_pnorm, *rope_cos, *rope_sin;
    struct {
        qmat_t qkv, proj, gate, up, down, ple_gate, ple_proj;
        const float *n1, *n2, *ple_norm;
        const uint8_t *downT;   // ReLU models: codes of down transposed, [ffn][d] (optional)
    } L[LLM_MAXL];
    const uint32_t *vocab_off;
    const uint8_t *vocab_bytes;
    // test section
    int n_prompts, n_gen, n_eval;
    const uint8_t *tests;
} llm_model_t;

// y[r*groups + g] = sum over the 32*bpg weights of group g of (nibble - 8) * x[j]
typedef void (*gemv_q4g_fn)(const uint8_t *W, const int8_t *x, int32_t *y, int rows, int groups, int bpg);
// y[r] = sum_{j < 16*n16} W[r*stride + j] * x[j]
typedef void (*gemv_s8_fn)(const int8_t *W, const int8_t *x, int32_t *y, int rows, int n16, int stride);
// out[l] = sum_{k<n} z[k] * (nibble l of the 16-byte block at rows[k]+off - 8), l < 16;
// half 0 = low nibbles (outputs j), 1 = high nibbles (outputs 16+j) of the block
typedef void (*colsum_fn)(const uint8_t *const *rows, const int8_t *z, int n, int off, int half, int32_t *out);
// run fn over [0, n): worker w gets its own slice; returns when all are done
typedef void (*par_body_fn)(void *ctx, int lo, int hi, int worker);

typedef struct {
    gemv_q4g_fn q4g;
    gemv_s8_fn s8;
    colsum_fn colsum;                                 // NULL: ReLU models read down in full
    void (*par)(par_body_fn fn, void *ctx, int n);   // NULL: run serially
    uint32_t (*cycles)(void);                         // NULL: no profiling
    // Called by llm_load for every tensor: return where its bytes will live
    // (copy to fast memory, or return src to use in place). kind 0 = q4 data,
    // 1 = q4 scales, 2 = f32.
    const void *(*place)(const char *name, const void *src, size_t n, int kind);
} llm_platform_t;

typedef struct llm llm_t;

int llm_load(llm_model_t *m, const uint8_t *blob, size_t size, const llm_platform_t *p);
llm_t *llm_new(const llm_model_t *m, const llm_platform_t *p,
               void *(*alloc)(size_t n, int fast));   // fast: internal SRAM preferred
void llm_reset(llm_t *s);
// One decode step: token at position pos (0-based). Returns vocab logits, or NULL if
// pos is outside [0, ctx) or the token outside [0, vocab). Positions may go back
// (rewind): a step at pos uses the KV cache of positions < pos only.
const float *llm_forward(llm_t *s, int token, int pos);
// n known tokens (prompt, schema-forced span) at positions pos..pos+n-1, n <= LLM_MAXB,
// weights read once. Bit-identical to n llm_forward calls; returns the last token's logits
// (NULL on the same conditions as llm_forward, for any of the n tokens).
const float *llm_forward_batch(llm_t *s, const int *tokens, int pos, int n);
// Two-stage head: logit of every token whose row was not read (never wins argmax or top-k).
#define LLM_UNREAD (-1e30f)
// Fraction of head rows to read per token (two-stage head); >= 1 or <= 0 reads all.
void llm_set_head_budget(llm_t *s, float budget);
// Experiment switches; none changes results.
#define LLM_DBG_WARM_KV 1   // touch each layer's KV cache lines before attention (timed as att_warm)
void llm_set_debug(llm_t *s, int flags);
// Recompute the head for the last forward's hidden state with another budget
// (into a second buffer); *rows gets the number of head rows read.
const float *llm_rehead(llm_t *s, float budget, int *rows);
// Cycles per stage accumulated since the last reset (ST_*).
const uint64_t *llm_stage_cycles(llm_t *s);
void llm_set_platform(llm_t *s, const llm_platform_t *p);

// deterministic math (no libm): identical bits on every IEEE-754 target
float llm_exp(float x);
float llm_log(float x);
float llm_recip(float x);
float llm_rsqrt(float x);
int llm_argmax(const float *v, int n);

// scalar reference kernels (the PIE kernels must match them bit for bit)
void llm_ref_q4g(const uint8_t *W, const int8_t *x, int32_t *y, int rows, int groups, int bpg);
void llm_ref_s8(const int8_t *W, const int8_t *x, int32_t *y, int rows, int n16, int stride);
void llm_ref_colsum(const uint8_t *const *rows, const int8_t *z, int n, int off, int half, int32_t *out);

// test runner shared by the host and the board: prints "ENG {json}" lines
typedef struct {
    const char *label;
    int prompts;       // how many prompts to generate from (<= n_prompts)
    int gen;           // tokens to generate per prompt (<= n_gen)
    int eval;          // teacher-forced tokens (<= n_eval), 0 = skip
    int repeats;       // timing repeats of prompt 0
    float head_budget; // gen/speed head budget (eval always reads the full head)
    int batch;         // >0: time these many eval tokens as known tokens in batches of 1, 2, .. LLM_MAXB
    int32_t *argmax_out; // optional: argmax per eval position
    float *logits_out;   // optional: all logits of the first eval window (ctx x vout)
    int rewind;          // 1: check that stepping back to an earlier position reproduces the logits
    int poscurve;        // 1: ms/token by context position, prompt 0 greedy up to ctx (needs cycles)
} llm_run_t;
void llm_run_tests(llm_t *s, const llm_model_t *m, const llm_run_t *r,
                   int (*out)(const char *fmt, ...), double cycles_per_us);
