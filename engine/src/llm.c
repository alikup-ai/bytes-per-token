// Portable decoder: see llm.h and engine/DESIGN.md.
//
// Every matvec is int4 weights x int8 activations. The platform kernel returns
// one exact int32 per 64-weight group; the float epilogue here folds in the
// group scale and the +0.5 level offset of the quantizer:
//   w = (n - 8 + 0.5) * s   =>   w . x = sa * s * (2 * sum((n-8) q) + sum(q)) / 2
// Float work uses only + - * and the functions below, in a fixed order, so the
// PC and the ESP32-S3 produce identical bits.
#include "llm.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

// ---------------------------------------------------------------- math ----
typedef union { float f; uint32_t u; int32_t i; } fu_t;

// Hot-path versions, inlined so the compiler can overlap independent chains:
// the S3 FPU has multi-cycle latency, so dependent-op depth is what costs
// (measured on the board: 4-step recip 115 cycles, Horner exp 174 cycles).
static inline float f_recip(float x) {       // x > 0, in [1e-30, 1e30]
    fu_t v = {.f = x};
    v.u = 0x7EF311C3u - v.u;                 // initial guess, rel. error <= 12%
    float y = v.f;
    y = y * (2.0f - x * y);                  // 3 Newton steps: error 12% -> 1.4% -> 2e-4 -> 4e-8
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}

static inline float f_rsqrt(float x) {       // x > 0
    fu_t v = {.f = x};
    v.u = 0x5F375A86u - (v.u >> 1);          // rel. error <= 0.2%
    float y = v.f, h = 0.5f * x;
    y = y * (1.5f - h * y * y);              // 3 steps: 0.2% -> 5e-6 -> 4e-11 (float-limited)
    y = y * (1.5f - h * y * y);
    y = y * (1.5f - h * y * y);
    return y;
}

static inline float f_exp(float x) {
    if (x < -87.0f) return 0.0f;
    if (x > 88.0f) x = 88.0f;
    float t = x * 1.44269504f + 0.5f;
    int n = (int)t;
    if ((float)n > t) n--;                               // floor
    float r = x - (float)n * 0.693145752f;               // ln2, high part (exact product)
    r = r - (float)n * 1.42860677e-6f;                   // ln2, low part
    // Taylor to r^6 on |r| <= ln2/2, Estrin order: depth 5 instead of Horner's 12
    float r2 = r * r, r4 = r2 * r2;
    float p = (1.0f + r) + r2 * (0.5f + r * 0.166666672f)
              + r4 * ((0.0416666679f + r * 0.00833333377f) + r2 * 0.00138888892f);
    fu_t v;
    v.i = (n + 127) << 23;
    return p * v.f;
}

float llm_recip(float x) { return x < 0.0f ? -f_recip(-x) : f_recip(x); }
float llm_rsqrt(float x) { return f_rsqrt(x); }
float llm_exp(float x) { return f_exp(x); }

float llm_log(float x) {             // x > 0, normal
    fu_t v = {.f = x};
    int e = (int)((v.u >> 23) & 255u) - 127;
    v.u = (v.u & 0x007FFFFFu) | 0x3F800000u;
    float m = v.f;
    if (m > 1.41421356f) { m = m * 0.5f; e++; }
    float f = m - 1.0f;
    float s = f * f_recip(2.0f + f);
    float z = s * s;
    float p = s * (2.0f + z * (0.666666667f + z * (0.4f + z * (0.285714286f + z * 0.222222222f))));
    return (float)e * 0.693147181f + p;
}

static float silu(float x) {
    if (x < -20.0f) return x * f_exp(x);
    return x * f_recip(1.0f + f_exp(-x));
}

static float gelu(float x) {         // erf by Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7)
    float z = x * 0.707106781f;
    float az = z < 0.0f ? -z : z;
    float t = f_recip(1.0f + 0.3275911f * az);
    float y = 1.0f - (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t
                      + 0.254829592f) * t * f_exp(-az * az);
    return 0.5f * x * (1.0f + (z < 0.0f ? -y : y));
}

// Activation tables, filled once from gelu()/silu() above (same bits on every
// target). Linear interpolation on [-LUT_R, LUT_R]; outside: 0 below, x above.
// Max error ~2e-5 (gelu, h = 1/64) and ~6e-5 (silu, h = 1/32), far below the
// 1/127 step of the int8 quantization that follows.
#define LUT_N 1024
typedef struct { float lo, scale, t[LUT_N + 1]; } lut_t;

static void lut_fill(lut_t *L, float r, float (*f)(float)) {
    L->lo = -r;
    L->scale = (float)LUT_N / (2.0f * r);
    for (int i = 0; i <= LUT_N; i++) L->t[i] = f(-r + (float)i * (2.0f * r / (float)LUT_N));
}

static inline float lut(const lut_t *L, float x) {
    float p = (x - L->lo) * L->scale;
    if (p <= 0.0f) return 0.0f;
    if (p >= (float)LUT_N) return x;
    int i = (int)p;
    float f = p - (float)i;
    return L->t[i] + f * (L->t[i + 1] - L->t[i]);
}

int llm_argmax(const float *v, int n) {
    int b = 0;
    for (int i = 1; i < n; i++)
        if (v[i] > v[b]) b = i;
    return b;
}

// ----------------------------------------------------- reference kernels ----
void llm_ref_q4g(const uint8_t *W, const int8_t *x, int32_t *y, int rows, int groups, int bpg) {
    for (int r = 0; r < rows; r++)
        for (int g = 0; g < groups; g++) {
            const uint8_t *w = W + ((size_t)r * groups + g) * bpg * 16;
            const int8_t *xg = x + g * bpg * 32;
            int32_t acc = 0;
            for (int b = 0; b < bpg; b++)
                for (int j = 0; j < 16; j++) {
                    uint8_t byte = w[b * 16 + j];
                    acc += ((int)(byte & 15) - 8) * xg[b * 32 + j] + ((int)(byte >> 4) - 8) * xg[b * 32 + 16 + j];
                }
            y[r * groups + g] = acc;
        }
}

void llm_ref_s8(const int8_t *W, const int8_t *x, int32_t *y, int rows, int n16, int stride) {
    for (int r = 0; r < rows; r++) {
        const int8_t *w = W + (size_t)r * stride;
        int32_t acc = 0;
        for (int j = 0; j < 16 * n16; j++) acc += w[j] * x[j];
        y[r] = acc;
    }
}

void llm_ref_colsum(const uint8_t *const *rows, const int8_t *z, int n, int off, int half, int32_t *out) {
    for (int l = 0; l < 16; l++) out[l] = 0;
    for (int k = 0; k < n; k++)
        for (int l = 0; l < 16; l++) {
            uint8_t b = rows[k][off + l];
            out[l] += z[k] * ((int)(half ? b >> 4 : b & 15) - 8);
        }
}

// ----------------------------------------------------------------- state ----
struct llm {
    const llm_model_t *m;
    llm_platform_t p;
    float *x, *h, *qkv, *att, *g, *u, *z, *tmp, *pl, *tab, *logits;
    // Batch slots: the per-token vectors of token t of a batch. Slot 0 is the
    // SRAM set above; slots 1.. live in PSRAM. use_slot() points x/qkv/att/
    // xq/A/live at a slot, so every kernel and helper works unchanged.
    struct slot { float *x, *qkv, *att, *g, *u, *z, *pl, h; int8_t *xq; int32_t *A; int *live, nlive; } sl[LLM_MAXB];
    int8_t *xq;
    int32_t *A, *part;
    int32_t *hpart[2];       // two-stage head: group sums of one cluster, per worker (not the whole vocabulary)
    int *live;
    const uint8_t **dptr;    // sparse down: row pointer and code per live neuron,
    int8_t *dz;              // grouped by 64-neuron group (live is sorted)
    int dgrp[LLM_MAXG + 1];  // live-list start of each group
    int8_t *K, *VT;
    int8_t *VS;      // V^T of one 16-position block per (layer, head), in SRAM: [L*H][dh][16]
    int *vsb;        // first position of the block VS holds, per (layer, head); -1 = none
    int hslot;       // batch slot whose xq/A fed the last head (llm_rehead reuses it)
    float *ks, *vs;
    struct { int8_t *qq, *cq, *vq; int32_t *sc, *o, *o2; float *score; } scr[2];
    uint64_t st[ST_N];
    uint64_t ast[4];
    uint64_t pc[7];          // per-token work by kind, core 0 (see PN in print_stages)
    lut_t *lgelu, *lsilu;    // activation tables (engine buffers)
    int dbg;                // LLM_DBG_* experiment switches (results unchanged)
    float hb, hh;            // head budget; epilogue factor of the last head input
    float *cs, *logits2;     // cluster scores; second logits buffer (llm_rehead)
    int *sel, *heap;         // selected clusters; selection heap
    uint8_t *mark;
    uint64_t head_rows;      // rows read, since reset   // attention sub-steps, worker 0 only: rope+store, scores, softmax, V product
    int pos;
};

// Symmetric int8 with one scale per vector. v3 (per-token work pass):
// |x| maximum found on the IEEE bit patterns (for non-negative floats integer
// order = float order), and round-to-nearest-even by adding 1.5 * 2^23, so
// the loop has no float->int conversion and no branches. |x * 127/amax| <= 127
// by construction, so no clamp is needed.
static float quant8(const float *x, int8_t *q, int n) {
    const uint32_t *u = (const uint32_t *)x;
    uint32_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        uint32_t a0 = u[i] & 0x7FFFFFFFu, a1 = u[i + 1] & 0x7FFFFFFFu;
        uint32_t a2 = u[i + 2] & 0x7FFFFFFFu, a3 = u[i + 3] & 0x7FFFFFFFu;
        m0 = a0 > m0 ? a0 : m0; m1 = a1 > m1 ? a1 : m1;
        m2 = a2 > m2 ? a2 : m2; m3 = a3 > m3 ? a3 : m3;
    }
    for (; i < n; i++) { uint32_t a = u[i] & 0x7FFFFFFFu; m0 = a > m0 ? a : m0; }
    m0 = m0 > m1 ? m0 : m1; m2 = m2 > m3 ? m2 : m3; m0 = m0 > m2 ? m0 : m2;
    if (m0 == 0) { memset(q, 0, (size_t)n); return 0.0f; }
    fu_t am = {.u = m0};
    float inv = 127.0f * f_recip(am.f);
    for (i = 0; i + 3 < n; i += 4) {
        fu_t r0 = {.f = x[i] * inv + 12582912.0f}, r1 = {.f = x[i + 1] * inv + 12582912.0f};
        fu_t r2 = {.f = x[i + 2] * inv + 12582912.0f}, r3 = {.f = x[i + 3] * inv + 12582912.0f};
        q[i] = (int8_t)(r0.i - 0x4B400000); q[i + 1] = (int8_t)(r1.i - 0x4B400000);
        q[i + 2] = (int8_t)(r2.i - 0x4B400000); q[i + 3] = (int8_t)(r3.i - 0x4B400000);
    }
    for (; i < n; i++) { fu_t r = {.f = x[i] * inv + 12582912.0f}; q[i] = (int8_t)(r.i - 0x4B400000); }
    return am.f * (1.0f / 127.0f);
}

// Quantize the input of matrix q into s->xq / s->A (code sums per scale group);
// returns sa/2 for the epilogue.
static inline uint32_t clk0(llm_t *s) { return s->p.cycles ? s->p.cycles() : 0; }

static float prep(llm_t *s, const float *x, int n, const qmat_t *q) {
    uint32_t t0 = clk0(s);
    float sa = quant8(x, s->xq, n);
    int gl = q->bpg * 32;
    const int8_t *c = s->xq;
    for (int g = 0; g < n / gl; g++, c += gl) {
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (int j = 0; j < gl; j += 4) { a0 += c[j]; a1 += c[j + 1]; a2 += c[j + 2]; a3 += c[j + 3]; }
        s->A[g] = a0 + a1 + a2 + a3;
    }
    s->pc[0] += (uint32_t)(clk0(s) - t0);
    return sa * 0.5f;
}

// IEEE half -> float, integer ops only (lookup-table scales; no inf/nan expected).
static inline float h2f(uint16_t h) {
    uint32_t sgn = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 31u, mnt = h & 1023u;
    fu_t v;
    if (e == 0) { v.f = (float)mnt * 5.96046448e-8f; v.u |= sgn; return v.f; }
    v.u = sgn | ((e + 112u) << 23) | (mnt << 13);
    return v.f;
}

// Epilogue of one output row: fold the group scales (and the +0.5 level offset
// when lvl = 1) into the exact integer group sums.
static inline float row_out(const qmat_t *m, int r, const int32_t *pt, const int32_t *A) {
    const float *sc = m->s + (size_t)r * m->groups;
    float acc = 0.0f;
    if (m->lvl)
        for (int g = 0; g < m->groups; g++) acc += sc[g] * (float)(2 * pt[g] + A[g]);
    else
        for (int g = 0; g < m->groups; g++) acc += sc[g] * (float)(2 * pt[g]);
    return acc;
}

static void rmsnorm(const float *x, const float *w, float *out, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = f_rsqrt(ss * f_recip((float)n) + 1e-6f);
    for (int i = 0; i < n; i++) out[i] = w[i] * x[i] * r;
}

static void dequant_row(const qmat_t *q, int r, float *out) {
    int gl = q->bpg * 32, off = q->lvl ? 15 : 16;       // w = (2*code - off) * s / 2
    for (int g = 0; g < q->groups; g++) {
        size_t gi = (size_t)r * q->groups + g;
        float hs = 0.5f * (q->s16 ? h2f(q->s16[gi]) : q->s[gi]);
        const uint8_t *w = q->w + gi * q->bpg * 16;
        for (int b = 0; b < q->bpg; b++)
            for (int j = 0; j < 16; j++) {
                uint8_t byte = w[b * 16 + j];
                out[g * gl + b * 32 + j] = (float)(2 * (int)(byte & 15) - off) * hs;
                out[g * gl + b * 32 + 16 + j] = (float)(2 * (int)(byte >> 4) - off) * hs;
            }
    }
}

// ---------------------------------------------------------------- matvec ----
typedef struct { const qmat_t *m; float h; float *out; int32_t *part; const int *idx; int n; } mv_t;
typedef struct { llm_t *s; mv_t j[2]; int nj; } mvjob_t;

static void mv_rows(llm_t *s, const mv_t *j, int lo, int hi) {
    const qmat_t *m = j->m;
    int G = m->groups;
    size_t rb = (size_t)G * m->bpg * 16;
    if (!j->idx) {
        s->p.q4g(m->w + (size_t)lo * rb, s->xq, j->part + (size_t)lo * G, hi - lo, G, m->bpg);
    }
    for (int k = lo; k < hi; k++) {
        int r = j->idx ? j->idx[k] : k;
        int32_t *pt = j->part + (size_t)k * G;
        if (j->idx) s->p.q4g(m->w + (size_t)r * rb, s->xq, pt, 1, G, m->bpg);
        j->out[r] = row_out(m, r, pt, s->A) * j->h;
    }
}

static void mv_body(void *c, int lo, int hi, int worker) {
    (void)worker;
    mvjob_t *J = c;
    int base = 0;
    for (int i = 0; i < J->nj; i++) {
        int a = lo > base ? lo : base, b = hi < base + J->j[i].n ? hi : base + J->j[i].n;
        if (a < b) mv_rows(J->s, &J->j[i], a - base, b - base);
        base += J->j[i].n;
    }
}

static void run(llm_t *s, par_body_fn fn, void *ctx, int n) {
    if (s->p.par) s->p.par(fn, ctx, n);
    else fn(ctx, 0, n, 0);
}

static void matvec(llm_t *s, const qmat_t *m, float h, float *out) {
    uint32_t t0 = clk0(s);
    mvjob_t J = {.s = s, .nj = 1};
    J.j[0] = (mv_t){m, h, out, s->part, NULL, m->rows};
    run(s, mv_body, &J, m->rows);
    s->pc[2] += (uint32_t)(clk0(s) - t0);
}

// ------------------------------------------------------- sparse down ----
// ReLU: z is zero outside the live neurons, so down's integer group sums only
// need the live columns. downT holds them as rows; colsum accumulates 16
// outputs at a time over the live rows of one group. The sums are the same
// integers the dense kernel produces, and the epilogue is the same code, so
// the result is bit-identical to reading down in full.
typedef struct { llm_t *s; const qmat_t *m; float h; float *out; } sdjob_t;

static void sdown_body(void *c, int lo, int hi, int worker) {
    (void)worker;
    sdjob_t *J = c;
    llm_t *s = J->s;
    const qmat_t *m = J->m;
    int G = m->groups;
    int32_t lane[16];
    for (int ch = lo; ch < hi; ch++) {                 // chunk = 16 outputs: block ch/2, half ch%2
        int b = ch >> 1, half = ch & 1, r0 = b * 32 + half * 16;
        for (int g = 0; g < G; g++) {
            int k0 = s->dgrp[g], nk = s->dgrp[g + 1] - k0;
            if (nk) s->p.colsum(s->dptr + k0, s->dz + k0, nk, b * 16, half, lane);
            for (int l = 0; l < 16; l++) s->part[(size_t)(r0 + l) * G + g] = nk ? lane[l] : 0;
        }
        for (int r = r0; r < r0 + 16; r++) J->out[r] = row_out(m, r, s->part + (size_t)r * G, s->A) * J->h;
    }
}

// ------------------------------------------------------------- attention ----
typedef struct { llm_t *s; int l, pos; } attjob_t;

static void att_body(void *c, int lo, int hi, int worker) {
    attjob_t *J = c;
    llm_t *s = J->s;
    const llm_model_t *m = s->m;
    int d = m->d, H = m->heads, dh = d / H, half = dh / 2, ctx = m->ctx, pos = J->pos, l = J->l;
    int ks16 = (dh + 15) & ~15;                       // K row stride: PIE reads 16 at a time, the pad stays 0
    const float *cs = m->rope_cos + (size_t)pos * half, *sn = m->rope_sin + (size_t)pos * half;
    float inv_sqrt_dh = f_rsqrt((float)dh);
    int n = pos + 1, n16 = (n + 15) / 16;
    uint32_t (*cy)(void) = worker == 0 ? s->p.cycles : NULL;
#define ASUB(i) do { if (cy) { uint32_t _t = cy(); s->ast[i] += (uint32_t)(_t - ta); ta = _t; } } while (0)
    for (int hh = lo; hh < hi; hh++) {
        uint32_t ta = cy ? cy() : 0;
        float *q = s->qkv + hh * dh, *k = s->qkv + d + hh * dh, *v = s->qkv + 2 * d + hh * dh;
        for (int i = 0; i < half; i++) {
            float a = q[i], b = q[i + half];
            q[i] = a * cs[i] - b * sn[i];
            q[i + half] = b * cs[i] + a * sn[i];
            a = k[i]; b = k[i + half];
            k[i] = a * cs[i] - b * sn[i];
            k[i + half] = b * cs[i] + a * sn[i];
        }
        size_t lh = (size_t)l * H + hh;
        int8_t *Kh = s->K + lh * ctx * ks16, *Vh = s->VT + lh * dh * ctx;
        float *ksh = s->ks + lh * ctx, *vsh = s->vs + lh * ctx;
        ksh[pos] = quant8(k, Kh + (size_t)pos * ks16, dh);
        int8_t *vq = s->scr[worker].vq;
        vsh[pos] = quant8(v, vq, dh);
        // V^T gets one byte per row per token: 32 scattered PSRAM cache lines.
        // Stage one 16-position block in SRAM. Entering another block (the next
        // one, or an earlier one after a rewind) writes the staged block back and
        // loads the new one, so VS always holds this block's positions < pos.
        int8_t *VSh = s->VS + lh * dh * 16;
        int blk = pos & ~15;
        if (s->vsb[lh] != blk) {
            if (s->vsb[lh] >= 0)
                for (int j = 0; j < dh; j++) memcpy(Vh + (size_t)j * ctx + s->vsb[lh], VSh + j * 16, 16);
            if (pos & 15)                                 // at a block start nothing before pos is in it
                for (int j = 0; j < dh; j++) memcpy(VSh + j * 16, Vh + (size_t)j * ctx + blk, 16);
            s->vsb[lh] = blk;
        }
        for (int j = 0; j < dh; j++) VSh[j * 16 + (pos & 15)] = vq[j];

        ASUB(0);
        int8_t *qq = s->scr[worker].qq, *cq = s->scr[worker].cq;
        int32_t *sc = s->scr[worker].sc, *o = s->scr[worker].o;
        float *score = s->scr[worker].score;
        float sq = quant8(q, qq, dh) * inv_sqrt_dh;
        s->p.s8(Kh, qq, sc, n, ks16 / 16, ks16);
        float mx = -1e30f;
        for (int t = 0; t < n; t++) {
            score[t] = (float)sc[t] * ksh[t] * sq;
            if (score[t] > mx) mx = score[t];
        }
        ASUB(1);
        float sum = 0.0f;
        int t = 0;
        for (; t + 1 < n; t += 2) {
            float e0 = f_exp(score[t] - mx), e1 = f_exp(score[t + 1] - mx);
            score[t] = e0; score[t + 1] = e1;
            sum += e0;
            sum += e1;
        }
        for (; t < n; t++) { score[t] = f_exp(score[t] - mx); sum += score[t]; }
        float inv = f_recip(sum);
        for (int t = 0; t < n; t++) score[t] = score[t] * inv * vsh[t];   // p_t times V scale
        float scq = quant8(score, cq, n);
        ASUB(2);
        memset(cq + n, 0, (size_t)(n16 * 16 - n));
        // Integer sums split at the block edge: identical to one pass over [0, n16*16).
        int32_t *o2 = s->scr[worker].o2;
        s->p.s8(VSh, cq + blk, o2, dh, 1, 16);
        if (blk) {
            s->p.s8(Vh, cq, o, dh, blk / 16, ctx);
            for (int j = 0; j < dh; j++) o[j] += o2[j];
        } else {
            for (int j = 0; j < dh; j++) o[j] = o2[j];
        }
        for (int j = 0; j < dh; j++) s->att[hh * dh + j] = (float)o[j] * scq;
        ASUB(3);
    }
#undef ASUB
}

// --------------------------------------------------------------- forward ----
static inline uint32_t clk(llm_t *s) { return s->p.cycles ? s->p.cycles() : 0; }
#define STAGE(id) do { uint32_t _t = clk(s); s->st[id] += (uint32_t)(_t - t0); t0 = _t; } while (0)

// ------------------------------------------------------------------ head ----
typedef struct { llm_t *s; float *out; } headjob_t;

static void head_body(void *c, int lo, int hi, int worker) {
    headjob_t *J = c;
    llm_t *s = J->s;
    const qmat_t *m = &s->m->head;
    int G = m->groups;
    for (int k = lo; k < hi; k++) {
        int cl = s->sel[k], r0 = (int)s->m->head_off[cl], r1 = (int)s->m->head_off[cl + 1];
        if (r1 <= r0) continue;
        int32_t *pt = s->hpart[worker];
        s->p.q4g(m->w + (size_t)r0 * G * m->bpg * 16, s->xq, pt, r1 - r0, G, m->bpg);
        for (int r = r0; r < r1; r++)
            J->out[s->m->head_perm[r]] = row_out(m, r, pt + (size_t)(r - r0) * G, s->A) * s->hh;
    }
}

static inline int before(const float *v, int a, int b) { return v[a] > v[b] || (v[a] == v[b] && a < b); }

static void sift(const float *v, int *h, int n, int i) {
    for (;;) {
        int l = 2 * i + 1, r = l + 1, t = i;
        if (l < n && before(v, h[l], h[t])) t = l;
        if (r < n && before(v, h[r], h[t])) t = r;
        if (t == i) return;
        int x = h[i]; h[i] = h[t]; h[t] = x;
        i = t;
    }
}

// Logits from s->xq / s->A (already prepared). Read rows get exactly the
// full-head value; unread rows get LLM_UNREAD, so neither argmax nor top-k
// sampling can pick a token whose row was not read. (Until 2026-09-29 they got
// their cluster's int4 centroid score: not a bound on the cluster's rows, and
// equal for a whole cluster, so top-k sampling could take unscored tokens.)
static int head(llm_t *s, float budget, float *out) {
    const llm_model_t *m = s->m;
    if (!m->clusters) {
        qmat_t h = m->emb;
        h.rows = m->vout;
        matvec(s, &h, s->hh, out);
        return m->vout;
    }
    int C = m->clusters, nsel = 0, rows = 0;
    if (budget <= 0.0f || budget >= 1.0f) {
        for (int c = 0; c < C; c++) s->sel[nsel++] = c;
        rows = m->vout;
    } else {
        matvec(s, &m->head_cent, s->hh, s->cs);
        uint32_t tsel = clk0(s);
        memset(s->mark, 0, (size_t)C);
        int want = (int)(budget * (float)m->vout);
        // Best clusters first (ties -> lower index) via a binary max-heap:
        // O(C + k log C) instead of k scans of all C scores.
        int *hp = s->heap, n = C;
        for (int c = 0; c < C; c++) hp[c] = c;
        for (int i = C / 2 - 1; i >= 0; i--) sift(s->cs, hp, n, i);
        while (rows < want && n > 0) {
            int b = hp[0];
            hp[0] = hp[--n];
            sift(s->cs, hp, n, 0);
            s->mark[b] = 1;
            s->sel[nsel++] = b;
            rows += (int)(m->head_off[b + 1] - m->head_off[b]);
        }
        for (int t = 0; t < m->vout; t++) out[t] = LLM_UNREAD;   // read rows overwritten below
        s->pc[4] += (uint32_t)(clk0(s) - tsel);
    }
    headjob_t J = {s, out};
    uint32_t th = clk0(s);
    run(s, head_body, &J, nsel);
    s->pc[2] += (uint32_t)(clk0(s) - th);
    return rows;
}

// Read every cache line attention will touch for layer l (K rows, V^T rows,
// K/V scales), so the attention that follows runs on a hot cache. Its cost is
// what PSRAM latency adds to attention.
static volatile uint32_t g_warm_sink;

static void warm_kv(llm_t *s, int l, int pos) {
    const llm_model_t *m = s->m;
    int H = m->heads, dh = m->d / H, ctx = m->ctx, ks16 = (dh + 15) & ~15, blk = pos & ~15;
    uint32_t acc = 0;
    for (int hh = 0; hh < H; hh++) {
        size_t lh = (size_t)l * H + hh;
        const uint8_t *K = (const uint8_t *)(s->K + lh * ctx * ks16);
        for (int o = 0; o < pos * ks16; o += 64) acc += K[o];
        const uint8_t *V = (const uint8_t *)(s->VT + lh * dh * ctx);
        for (int j = 0; j < dh; j++)
            for (int o = 0; o < blk; o += 64) acc += V[(size_t)j * ctx + o];
        const uint8_t *ks = (const uint8_t *)(s->ks + lh * ctx), *vs = (const uint8_t *)(s->vs + lh * ctx);
        for (int o = 0; o < pos * 4; o += 64) acc += ks[o] + vs[o];
    }
    g_warm_sink = acc;
}

#define PT(k, stmt) do { uint32_t _a = clk0(s); stmt; s->pc[k] += (uint32_t)(clk0(s) - _a); } while (0)

static inline void use_slot(llm_t *s, int t) {
    struct slot *q = &s->sl[t];
    s->x = q->x; s->qkv = q->qkv; s->att = q->att; s->g = q->g; s->u = q->u; s->z = q->z; s->pl = q->pl;
    s->xq = q->xq; s->A = q->A; s->live = q->live;
}

// n tokens at positions pos..pos+n-1, weights read once per batch: every
// phase loops over the tokens back to back, so after the first token a matrix
// (all < 64 KB) comes from the data cache. Each token gets exactly the
// operations of a single-token step, and attention runs token by token within
// a layer, so the results are bit-identical to n calls of llm_forward.
// Returns the logits of the last token only (prefill, forced tokens).
const float *llm_forward_batch(llm_t *s, const int *toks, int pos, int n) {
    const llm_model_t *m = s->m;
    int d = m->d, L = m->layers, P = m->ple, F = m->ffn;
    if (n < 1 || n > LLM_MAXB || pos < 0 || pos + n > m->ctx) return NULL;
    for (int t = 0; t < n; t++)
        if (toks[t] < 0 || toks[t] >= m->vocab) return NULL;
    uint32_t t0 = clk(s);

    for (int t = 0; t < n; t++) {
        use_slot(s, t);
        dequant_row(&m->emb, m->emb_row ? (int)m->emb_row[toks[t]] : toks[t], s->x);
    }
    if (P) {
        float isd = f_rsqrt((float)d), sp = (float)P * f_rsqrt((float)P);
        for (int t = 0; t < n; t++) {
            use_slot(s, t);
            matvec(s, &m->ple_mproj, prep(s, s->x, d, &m->ple_mproj), s->pl);
            dequant_row(&m->ple_table, toks[t], s->tab);
            for (int l = 0; l < L; l++) {
                float *pl = s->pl + l * P;
                PT(3, for (int j = 0; j < P; j++) pl[j] = pl[j] * isd);
                PT(1, rmsnorm(pl, m->ple_pnorm, pl, P));
                PT(3, for (int j = 0; j < P; j++) pl[j] = (pl[j] + s->tab[l * P + j] * sp) * 0.707106781f);
            }
        }
    }
    STAGE(ST_INPUT);

    for (int l = 0; l < L; l++) {
        for (int t = 0; t < n; t++) {
            use_slot(s, t);
            PT(1, rmsnorm(s->x, m->L[l].n1, s->h, d));
            matvec(s, &m->L[l].qkv, prep(s, s->h, d, &m->L[l].qkv), s->qkv);
        }
        STAGE(ST_ATTN_MV);
        for (int t = 0; t < n; t++) {                    // in position order: token t sees tokens < t
            use_slot(s, t);
            if (s->dbg & LLM_DBG_WARM_KV) {              // experiment: pull this layer's KV into the cache first
                uint32_t tw = clk0(s);
                warm_kv(s, l, pos + t);
                s->pc[6] += (uint32_t)(clk0(s) - tw);
                t0 = clk(s);                             // the attention stage then measures hot-cache compute
            }
            attjob_t aj = {s, l, pos + t};
            run(s, att_body, &aj, m->heads);
        }
        STAGE(ST_ATTN);
        for (int t = 0; t < n; t++) {
            use_slot(s, t);
            matvec(s, &m->L[l].proj, prep(s, s->att, d, &m->L[l].proj), s->tmp);
            PT(3, for (int i = 0; i < d; i++) s->x[i] += s->tmp[i]);
        }
        STAGE(ST_ATTN_MV);

        for (int t = 0; t < n; t++) {                    // gate (and up, SwiGLU) for every token
            use_slot(s, t);
            PT(1, rmsnorm(s->x, m->L[l].n2, s->h, d));
            float h = s->sl[t].h = prep(s, s->h, d, &m->L[l].gate);
            if (m->act == ACT_SILU) {
                mvjob_t J = {.s = s, .nj = 2};
                J.j[0] = (mv_t){&m->L[l].gate, h, s->g, s->part, NULL, F};
                J.j[1] = (mv_t){&m->L[l].up, h, s->u, s->part + (size_t)F * m->L[l].up.groups, NULL, F};
                PT(2, run(s, mv_body, &J, 2 * F));
                PT(3, for (int r = 0; r < F; r++) s->z[r] = lut(s->lsilu, s->g[r]) * s->u[r]);
            } else {
                matvec(s, &m->L[l].gate, h, s->g);
                int nl = 0;                               // ReLU gate-first: only live rows of up
                PT(5, for (int r = 0; r < F; r++) if (s->g[r] > 0.0f) s->live[nl++] = r);
                s->sl[t].nlive = nl;
            }
        }
        if (m->act != ACT_SILU)
            for (int t = 0; t < n; t++) {                // up on each token's live rows (xq/A of its gate input)
                use_slot(s, t);
                int nl = s->sl[t].nlive;
                mvjob_t J = {.s = s, .nj = 1};
                J.j[0] = (mv_t){&m->L[l].up, s->sl[t].h, s->u, s->part, s->live, nl};
                PT(2, run(s, mv_body, &J, nl));
                memset(s->z, 0, sizeof(float) * (size_t)F);
                for (int k = 0; k < nl; k++) {
                    int r = s->live[k];
                    float a = m->act == ACT_RELU2 ? s->g[r] * s->g[r] : s->g[r];
                    s->z[r] = a * s->u[r];
                }
            }
        for (int t = 0; t < n; t++) {
            use_slot(s, t);
            float hz = prep(s, s->z, F, &m->L[l].down);
            // the live list is bucketed by 64 neurons: only valid for 64-wide down groups
            if (m->act != ACT_SILU && m->L[l].downT && s->p.colsum && m->L[l].down.bpg == 2) {
                int nl = s->sl[t].nlive, rb = d / 2;
                uint32_t tb = clk0(s);
                for (int g = 0; g <= F / 64; g++) s->dgrp[g] = 0;
                for (int k = 0; k < nl; k++) {              // the live list of the up pass (sorted)
                    int r = s->live[k];
                    s->dptr[k] = m->L[l].downT + (size_t)r * rb;
                    s->dz[k] = s->xq[r];
                    s->dgrp[r / 64 + 1] = k + 1;
                }
                for (int g = 1; g <= F / 64; g++)         // groups with no live neuron end where the last one did
                    if (s->dgrp[g] < s->dgrp[g - 1]) s->dgrp[g] = s->dgrp[g - 1];
                s->pc[5] += (uint32_t)(clk0(s) - tb);
                sdjob_t J = {s, &m->L[l].down, hz, s->tmp};
                PT(2, run(s, sdown_body, &J, d / 16));
            } else {
                matvec(s, &m->L[l].down, hz, s->tmp);
            }
            PT(3, for (int i = 0; i < d; i++) s->x[i] += s->tmp[i]);
        }
        STAGE(ST_FFN);

        if (P) {
            for (int t = 0; t < n; t++) {
                use_slot(s, t);
                matvec(s, &m->L[l].ple_gate, prep(s, s->x, d, &m->L[l].ple_gate), s->g);
                PT(3, for (int j = 0; j < P; j++) s->g[j] = lut(s->lgelu, s->g[j]) * s->pl[l * P + j]);
                matvec(s, &m->L[l].ple_proj, prep(s, s->g, P, &m->L[l].ple_proj), s->tmp);
                PT(1, rmsnorm(s->tmp, m->L[l].ple_norm, s->tmp, d));
                PT(3, for (int i = 0; i < d; i++) s->x[i] += s->tmp[i]);
            }
            STAGE(ST_PLE);
        }
    }
    use_slot(s, n - 1);
    s->hslot = n - 1;
    PT(1, rmsnorm(s->x, m->norm, s->h, d));
    s->hh = prep(s, s->h, d, m->clusters ? &m->head : &m->emb);
    s->head_rows += (uint64_t)head(s, s->hb, s->logits);
    STAGE(ST_HEAD);
    use_slot(s, 0);
    return s->logits;
}

const float *llm_forward(llm_t *s, int tok, int pos) { return llm_forward_batch(s, &tok, pos, 1); }

// ------------------------------------------------------------ load / new ----
// e: kind rows cols off_w off_s [group length, pack v3]. kind 0: fp32 scales, 5: fp16 scales.
// [off, off + n) inside a blob of `size` bytes (no overflow: n is at most ~2^40 here)
static int inside(size_t size, uint64_t off, uint64_t n) { return off <= size && n <= size - off; }

static int set_q(qmat_t *q, const uint8_t *b, size_t size, const uint32_t *e, int gl, int lvl, const char *name,
                 const llm_platform_t *p) {
    if (gl <= 0 || gl % 32 || e[2] % (uint32_t)gl) return -7;
    q->rows = (int)e[1];
    q->cols = (int)e[2];
    q->groups = q->cols / gl;
    q->bpg = gl / 32;
    q->lvl = lvl;
    uint64_t nw = (uint64_t)e[1] * e[2] / 2, ns = (uint64_t)e[1] * (uint64_t)q->groups * (e[0] == 5 ? 2 : 4);
    if (!inside(size, e[3], nw) || !inside(size, e[4], ns)) return -6;
    q->w = p->place(name, b + e[3], (size_t)nw, 0);
    if (e[0] == 5) q->s16 = p->place(name, b + e[4], (size_t)ns, 1);
    else q->s = p->place(name, b + e[4], (size_t)ns, 1);
    return 0;
}

int llm_load(llm_model_t *m, const uint8_t *b, size_t size, const llm_platform_t *p) {
    // v1: our models (group 64, half-integer levels). v3 adds vout, levels and a
    // group length per tensor (engine/export_esp32ai.py).
    int v3 = size >= 64 && memcmp(b, "MCULM03", 8) == 0;
    if (size < 64 || (!v3 && memcmp(b, "MCULM01", 8) != 0)) return -1;
    const uint32_t *h = (const uint32_t *)(b + 8);
    memset(m, 0, sizeof *m);
    m->vocab = h[0]; m->d = h[1]; m->layers = h[2]; m->heads = h[3]; m->ffn = h[4];
    m->ple = h[5]; m->act = h[6]; m->ctx = h[7];
    uint32_t n = h[8], dir = h[9], voc = h[10], test = h[11], total = h[12];
    m->vout = v3 ? (int)h[13] : m->vocab;
    m->lvl = v3 ? (int)h[14] : 1;
    int esz = v3 ? 48 : 44;
    if (total > size || m->layers > LLM_MAXL || m->vout > m->vocab || m->vocab < 1 || m->ctx < 1) return -2;
    if (!inside(total, dir, (uint64_t)n * esz) || !inside(total, voc, 4 * ((uint64_t)m->vocab + 1)) ||
        !inside(total, test, 12))
        return -2;
    size = total;                                    // every tensor must lie inside the pack
    uint32_t n_emb_row = 0;
    for (uint32_t i = 0; i < n; i++) {
        const uint8_t *ent = b + dir + i * esz;
        char name[25];
        memcpy(name, ent, 24);
        name[24] = 0;
        const uint32_t *e = (const uint32_t *)(ent + 24);   // kind rows cols off_w off_s [group length]
        int gl = v3 ? (int)e[5] : 64;
        uint64_t rc = (uint64_t)e[1] * e[2];
        // byte extent of the non-q4 kinds: 1 f32 [rows][cols], 2 u32 [cols], 3 packed codes
        uint64_t nb = e[0] == 1 ? rc * 4 : e[0] == 2 ? (uint64_t)e[2] * 4 : e[0] == 3 ? rc / 2 : 0;
        if (nb && !inside(size, e[3], nb)) return -6;
        const float *f = e[0] == 1 ? p->place(name, b + e[3], (size_t)nb, 2) : NULL;
        const void *arr = e[0] == 2 ? p->place(name, b + e[3], (size_t)nb, 2) : NULL;
        int l = -1, err = 0;
        char what[24] = "";
        if (name[0] == 'L') sscanf(name, "L%d.%23s", &l, what);
#define Q(dst) err = set_q(&(dst), b, size, e, gl, m->lvl, name, p)
        if (l < 0) {
            if (!strcmp(name, "emb")) Q(m->emb);
            else if (!strcmp(name, "ple_table")) Q(m->ple_table);
            else if (!strcmp(name, "ple_mproj")) Q(m->ple_mproj);
            else if (!strcmp(name, "head")) Q(m->head);
            else if (!strcmp(name, "head_cent")) Q(m->head_cent);
            else if (!strcmp(name, "head_perm")) m->head_perm = arr;
            else if (!strcmp(name, "emb_row")) { m->emb_row = arr; n_emb_row = e[2]; }
            else if (!strcmp(name, "head_off")) {
                m->head_off = arr;
                m->clusters = (int)e[2] - 1;
            }
            else if (!strcmp(name, "norm")) m->norm = f;
            else if (!strcmp(name, "ple_pnorm")) m->ple_pnorm = f;
            else if (!strcmp(name, "rope_cos")) m->rope_cos = f;
            else if (!strcmp(name, "rope_sin")) m->rope_sin = f;
            else return -3;
            if (err) return err;
            continue;
        }
        if (l >= m->layers) return -4;
        if (!strcmp(what, "qkv")) Q(m->L[l].qkv);
        else if (!strcmp(what, "proj")) Q(m->L[l].proj);
        else if (!strcmp(what, "gate")) Q(m->L[l].gate);
        else if (!strcmp(what, "up")) Q(m->L[l].up);
        else if (!strcmp(what, "down")) Q(m->L[l].down);
        else if (!strcmp(what, "ple_gate")) Q(m->L[l].ple_gate);
        else if (!strcmp(what, "ple_proj")) Q(m->L[l].ple_proj);
        else if (!strcmp(what, "n1")) m->L[l].n1 = f;
        else if (!strcmp(what, "n2")) m->L[l].n2 = f;
        else if (!strcmp(what, "ple_norm")) m->L[l].ple_norm = f;
        else if (!strcmp(what, "downT")) m->L[l].downT = p->place(name, b + e[3], (size_t)nb, 0);
        else return -5;
#undef Q
        if (err) return err;
    }
    if (m->emb_row && m->clusters && !m->head.w) {   // single copy: the head is the first vout rows
        m->head = m->emb;
        m->head.rows = m->vout;
    }
    if (m->emb_row) {                                // token -> embedding row, for every input token
        if (n_emb_row < (uint32_t)m->vocab) return -8;
        for (int v = 0; v < m->vocab; v++)
            if (m->emb_row[v] >= (uint32_t)m->emb.rows) return -8;
    }
    // head_perm / head_off must cover the head, and cluster offsets must stay in range
    if (m->clusters) {
        if (!m->head_perm || !m->head_off || m->clusters < 1) return -8;
        for (int c = 0; c < m->clusters; c++)
            if (m->head_off[c] > m->head_off[c + 1] || m->head_off[c + 1] > (uint32_t)m->vout) return -8;
    }
    m->vocab_off = (const uint32_t *)(b + voc);
    m->vocab_bytes = b + voc + 4 * (m->vocab + 1);
    const uint32_t *t = (const uint32_t *)(b + test);
    m->n_prompts = (int)t[0]; m->n_gen = (int)t[1]; m->n_eval = (int)t[2];
    m->tests = b + test + 12;
    return 0;
}

#define MAX2(a, b) ((a) > (b) ? (a) : (b))

llm_t *llm_new(const llm_model_t *m, const llm_platform_t *p, void *(*alloc)(size_t, int)) {
    llm_t *s = alloc(sizeof(llm_t), 1);
    if (!s) return NULL;
    memset(s, 0, sizeof *s);
    s->m = m;
    s->p = *p;
    int d = m->d, F = m->ffn, P = m->ple, LP = m->layers * m->ple, dh = d / m->heads, ctx = m->ctx;
    int ks16 = (dh + 15) & ~15;
    int vec = MAX2(MAX2(d, F), MAX2(LP, 64));
    size_t part = 0;                                   // widest rows x groups of any matvec
#define PM(q, r) part = MAX2(part, (size_t)(r) * (size_t)(q).groups)
    if (m->clusters) PM(m->head_cent, m->clusters);   // the head itself uses hpart, one cluster at a time
    else PM(m->emb, m->vout);
    if (P) { PM(m->ple_mproj, LP); PM(m->L[0].ple_gate, P); PM(m->L[0].ple_proj, d); }
    PM(m->L[0].qkv, 3 * d); PM(m->L[0].proj, d); PM(m->L[0].gate, 2 * F); PM(m->L[0].down, d);
#undef PM
#define F32(n) ((float *)alloc(sizeof(float) * (size_t)(n), 1))
    s->x = F32(d); s->h = F32(d); s->qkv = F32(3 * d); s->att = F32(d); s->tmp = F32(d);
    s->g = F32(vec); s->u = F32(F); s->z = F32(F); s->pl = F32(MAX2(LP, 1)); s->tab = F32(MAX2(LP, 1));
    s->logits = F32(m->vout);
    if (m->clusters) {
        s->logits2 = alloc(sizeof(float) * (size_t)m->vout, 0);
        s->cs = F32(m->clusters);
        s->sel = alloc(sizeof(int) * (size_t)m->clusters, 1);
        s->heap = alloc(sizeof(int) * (size_t)m->clusters, 1);
        s->mark = alloc((size_t)m->clusters, 1);
        if (!s->logits2 || !s->cs || !s->sel || !s->heap || !s->mark) return NULL;
    }
    s->hb = 1.0f;
    for (int t = 0; t < LLM_MAXB; t++) {               // slot 0 = the SRAM buffers; others in PSRAM
        struct slot *q = &s->sl[t];
        int f = t == 0;
        q->x = t ? alloc(sizeof(float) * (size_t)d, f) : s->x;
        q->qkv = t ? alloc(sizeof(float) * (size_t)3 * d, f) : s->qkv;
        q->att = t ? alloc(sizeof(float) * (size_t)d, f) : s->att;
        q->g = t ? alloc(sizeof(float) * (size_t)vec, f) : s->g;
        q->u = t ? alloc(sizeof(float) * (size_t)F, f) : s->u;
        q->z = t ? alloc(sizeof(float) * (size_t)F, f) : s->z;
        q->pl = t ? alloc(sizeof(float) * (size_t)MAX2(LP, 1), f) : s->pl;
        if (!q->x || !q->qkv || !q->att || !q->g || !q->u || !q->z || !q->pl) return NULL;
    }
    s->lgelu = alloc(sizeof(lut_t), 1);
    s->lsilu = alloc(sizeof(lut_t), 1);
    if (!s->lgelu || !s->lsilu) return NULL;
    lut_fill(s->lgelu, 8.0f, gelu);
    lut_fill(s->lsilu, 16.0f, silu);
    s->xq = alloc((size_t)vec, 1);
    s->A = alloc(sizeof(int32_t) * (size_t)(vec / 32 + 1), 1);
    for (int t = 0; t < LLM_MAXB; t++) {
        s->sl[t].xq = t ? alloc((size_t)vec, 0) : s->xq;
        s->sl[t].A = t ? alloc(sizeof(int32_t) * (size_t)(vec / 32 + 1), 0) : s->A;
        if (!s->sl[t].xq || !s->sl[t].A) return NULL;
    }
    s->part = alloc(sizeof(int32_t) * part, 1);
    if (m->clusters) {                                  // largest cluster x groups, one buffer per worker
        size_t maxc = 0;
        for (int c = 0; c < m->clusters; c++) maxc = MAX2(maxc, (size_t)(m->head_off[c + 1] - m->head_off[c]));
        for (int w = 0; w < 2; w++) {
            s->hpart[w] = alloc(sizeof(int32_t) * MAX2(maxc, 1) * (size_t)m->head.groups, 1);
            if (!s->hpart[w]) return NULL;
        }
    }
    s->live = alloc(sizeof(int) * (size_t)F, 1);
    for (int t = 0; t < LLM_MAXB; t++) {
        s->sl[t].live = t ? alloc(sizeof(int) * (size_t)F, 0) : s->live;
        if (!s->sl[t].live) return NULL;
    }
    s->dptr = alloc(sizeof(uint8_t *) * (size_t)F, 1);
    s->dz = alloc((size_t)F, 1);
    if (F / 64 > LLM_MAXG) return NULL;
    size_t kv = (size_t)m->layers * m->heads * ctx;
    s->K = alloc(kv * ks16, 0); s->VT = alloc(kv * dh, 0);
    s->VS = alloc((size_t)m->layers * m->heads * dh * 16, 1);
    s->vsb = alloc(sizeof(int) * (size_t)m->layers * m->heads, 1);
    if (!s->VS || !s->vsb) return NULL;
    for (int i = 0; i < m->layers * m->heads; i++) s->vsb[i] = -1;
    s->ks = alloc(sizeof(float) * (size_t)m->layers * m->heads * ctx, 0);
    s->vs = alloc(sizeof(float) * (size_t)m->layers * m->heads * ctx, 0);
    for (int w = 0; w < 2; w++) {
        s->scr[w].qq = alloc((size_t)ks16, 1); s->scr[w].vq = alloc((size_t)dh, 1);   // qq pad stays 0
        s->scr[w].cq = alloc((size_t)ctx + 16, 1);
        s->scr[w].sc = alloc(sizeof(int32_t) * (size_t)ctx, 1);
        s->scr[w].o = alloc(sizeof(int32_t) * (size_t)dh, 1);
        s->scr[w].o2 = alloc(sizeof(int32_t) * (size_t)dh, 1);
        s->scr[w].score = F32(ctx);
    }
#undef F32
    if (!s->x || !s->logits || !s->part || !s->K || !s->VT || !s->scr[1].score) return NULL;
    return s;
}

void llm_set_head_budget(llm_t *s, float budget) { s->hb = budget; }
void llm_set_debug(llm_t *s, int flags) { s->dbg = flags; }

const float *llm_rehead(llm_t *s, float budget, int *rows) {
    if (!s->logits2) return NULL;
    use_slot(s, s->hslot);                       // the head input of the last forward (last token of a batch)
    int r = head(s, budget, s->logits2);
    use_slot(s, 0);
    if (rows) *rows = r;
    return s->logits2;
}

void llm_reset(llm_t *s) { s->head_rows = 0; memset(s->st, 0, sizeof s->st); memset(s->ast, 0, sizeof s->ast); memset(s->pc, 0, sizeof s->pc); }
const uint64_t *llm_stage_cycles(llm_t *s) { return s->st; }
void llm_set_platform(llm_t *s, const llm_platform_t *p) { s->p = *p; }

// ------------------------------------------------------------ test runner ----
static uint32_t fnv(uint32_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 16777619u; }
    return h;
}

static const uint16_t *prompt(const llm_model_t *m, int k, int *len) {
    const uint8_t *t = m->tests;
    for (int i = 0; i < k; i++) {
        uint32_t n;
        memcpy(&n, t, 4);
        t += 4 + 2 * n;
    }
    uint32_t n;
    memcpy(&n, t, 4);
    *len = (int)n;
    return (const uint16_t *)(t + 4);
}

static const uint16_t *eval_tokens(const llm_model_t *m) {
    int len;
    const uint16_t *p = prompt(m, m->n_prompts, &len);   // just past the last prompt
    return (const uint16_t *)((const uint8_t *)p - 4);
}

static const char *SN[ST_N] = {"input", "attn_mv", "attn", "ffn", "ple", "head"};

static void print_stages(llm_t *s, int (*out)(const char *, ...), double cpu, int ntok) {
    out(",\"stage_ms_per_tok\":{");
    for (int i = 0; i < ST_N; i++)
        out("%s\"%s\":%.3f", i ? "," : "", SN[i], (double)s->st[i] / cpu / 1000.0 / ntok);
    static const char *AN[4] = {"att_rope_store", "att_scores", "att_softmax", "att_v"};
    for (int i = 0; i < 4; i++) out(",\"%s\":%.3f", AN[i], (double)s->ast[i] / cpu / 1000.0 / ntok);
    static const char *PN[7] = {"prep", "norm", "matvec", "elem", "head_select", "sparse_book", "att_warm"};
    out("},\"work_ms_per_tok\":{");
    for (int i = 0; i < 7; i++) out("%s\"%s\":%.3f", i ? "," : "", PN[i], (double)s->pc[i] / cpu / 1000.0 / ntok);
    out("},\"head_rows_per_tok\":%.1f", (double)s->head_rows / ntok);
}

// Prompt of len tokens, in batches of up to LLM_MAXB; returns the last logits.
static const float *prefill(llm_t *s, const uint16_t *p, int len) {
    const float *lg = NULL;
    int tk[LLM_MAXB];
    for (int i = 0; i < len; i += LLM_MAXB) {
        int n = len - i < LLM_MAXB ? len - i : LLM_MAXB;
        for (int k = 0; k < n; k++) tk[k] = p[i + k];
        lg = llm_forward_batch(s, tk, i, n);
    }
    return lg;
}

void llm_run_tests(llm_t *s, const llm_model_t *m, const llm_run_t *r,
                   int (*out)(const char *, ...), double cpu) {
    int ids[512];
    llm_set_head_budget(s, r->head_budget);
    for (int k = 0; k < r->prompts; k++) {
        int len;
        const uint16_t *p = prompt(m, k, &len);
        uint32_t td = 2166136261u, ld = 2166136261u;
        const float *lg = NULL;
        uint32_t t0 = s->p.cycles ? s->p.cycles() : 0;
        lg = prefill(s, p, len);
        uint32_t t1 = s->p.cycles ? s->p.cycles() : 0;
        llm_reset(s);
        int n = 0;
        uint64_t gen_cyc = 0;
        for (int pos = len; n < r->gen && n < (int)(sizeof ids / sizeof ids[0]) && pos < m->ctx; pos++) {
            ld = fnv(ld, lg, sizeof(float) * (size_t)m->vout);
            int t = llm_argmax(lg, m->vout);
            ids[n++] = t;
            td = fnv(td, &t, sizeof t);
            uint32_t a = s->p.cycles ? s->p.cycles() : 0;
            lg = llm_forward(s, t, pos);
            gen_cyc += (uint32_t)((s->p.cycles ? s->p.cycles() : 0) - a);
        }
        out("ENG {\"label\":\"%s\",\"test\":\"gen\",\"prompt\":%d,\"ids\":[", r->label, k);
        for (int i = 0; i < n; i++) out("%s%d", i ? "," : "", ids[i]);
        out("],\"tok_digest\":\"%08x\",\"logit_digest\":\"%08x\",\"prefill_ms\":%.2f,\"ms_per_tok\":%.3f",
            td, ld, (double)(uint32_t)(t1 - t0) / cpu / 1000.0, (double)gen_cyc / cpu / 1000.0 / n);
        print_stages(s, out, cpu, n);
        out("}\n");
    }
    if (r->rewind) {
        // Go forward to position `last`, then step back into an earlier block
        // (mid-block, so the staged V^T block has to be written back and reloaded)
        // and redo a few positions: logits must equal the first pass bit for bit.
        int len, last = m->ctx - 1 < 60 ? m->ctx - 1 : 60, tk[64];
        uint32_t dg[64];
        const uint16_t *p = prompt(m, 0, &len);
        int at = (last & ~15) - 11, ok = 1, redo = 0;
        if (at >= len && at > 0 && len <= LLM_MAXB * 4) {
            const float *lg = prefill(s, p, len);
            for (int i = 0; i < len; i++) tk[i] = p[i];
            for (int pos = len; pos <= last && lg; pos++) {
                tk[pos] = llm_argmax(lg, m->vout);
                lg = llm_forward(s, tk[pos], pos);
                dg[pos] = lg ? fnv(2166136261u, lg, sizeof(float) * (size_t)m->vout) : 0;
            }
            for (int pos = at; pos <= last && pos < at + 5; pos++, redo++) {
                const float *l2 = llm_forward(s, tk[pos], pos);
                ok &= l2 && fnv(2166136261u, l2, sizeof(float) * (size_t)m->vout) == dg[pos];
            }
            out("ENG {\"label\":\"%s\",\"test\":\"rewind\",\"from\":%d,\"to\":%d,\"redone\":%d,\"ok\":%d}\n",
                r->label, last, at, redo, ok);
        }
    }
    if (r->poscurve && s->p.cycles) {
        // ms/token of greedy generation by context position, 32 positions per bucket
        enum { NB = 64 };
        uint64_t cyc[NB] = {0};
        int cnt[NB] = {0}, len;
        const uint16_t *p = prompt(m, 0, &len);
        const float *lg = prefill(s, p, len);
        for (int pos = len; pos < m->ctx && pos / 32 < NB && lg; pos++) {
            int t = llm_argmax(lg, m->vout);
            uint32_t a = s->p.cycles();
            lg = llm_forward(s, t, pos);
            cyc[pos / 32] += (uint32_t)(s->p.cycles() - a);
            cnt[pos / 32]++;
        }
        out("ENG {\"label\":\"%s\",\"test\":\"poscurve\",\"bucket\":32,\"first_pos\":%d,\"ms_per_tok\":[", r->label, len);
        for (int b = 0, first = 1; b < NB; b++)
            if (cnt[b]) { out("%s%.3f", first ? "" : ",", (double)cyc[b] / cpu / 1000.0 / cnt[b]); first = 0; }
        out("]}\n");
    }
    for (int rep = 0; rep < r->repeats; rep++) {
        int len;
        const uint16_t *p = prompt(m, 0, &len);
        const float *lg = prefill(s, p, len);
        llm_reset(s);
        uint64_t cyc = 0;
        int n = 0;
        for (int pos = len; n < r->gen && pos < m->ctx; pos++, n++) {
            int t = llm_argmax(lg, m->vout);
            uint32_t a = s->p.cycles ? s->p.cycles() : 0;
            lg = llm_forward(s, t, pos);
            cyc += (uint32_t)((s->p.cycles ? s->p.cycles() : 0) - a);
        }
        out("ENG {\"label\":\"%s\",\"test\":\"speed\",\"rep\":%d,\"tokens\":%d,\"ms_per_tok\":%.3f",
            r->label, rep, n, (double)cyc / cpu / 1000.0 / n);
        print_stages(s, out, cpu, n);
        out("}\n");
    }
    if (r->batch) {
        const uint16_t *ev = eval_tokens(m);
        int N = r->batch, tk[LLM_MAXB];
        out("ENG {\"label\":\"%s\",\"test\":\"batch\",\"tokens\":%d,\"runs\":[", r->label, N);
        for (int bs = 1, first = 1; bs <= LLM_MAXB; bs *= 2, first = 0) {
            const float *lg = NULL;
            uint32_t a = s->p.cycles ? s->p.cycles() : 0;
            for (int i = 0; i < N; i += bs) {
                int n = N - i < bs ? N - i : bs;
                for (int k = 0; k < n; k++) tk[k] = ev[i + k];
                lg = llm_forward_batch(s, tk, i, n);
            }
            uint32_t b = s->p.cycles ? s->p.cycles() : 0;
            uint32_t dg = fnv(2166136261u, lg, sizeof(float) * (size_t)m->vout);
            out("%s{\"batch\":%d,\"ms_per_tok\":%.3f,\"last_logits_digest\":\"%08x\"}", first ? "" : ",", bs,
                (double)(uint32_t)(b - a) / cpu / 1000.0 / N, dg);
        }
        out("]}\n");
    }
    if (r->eval > 0) {
        llm_set_head_budget(s, 1.0f);                      // NLL needs the full distribution
        int screen = m->clusters && r->head_budget > 0.0f && r->head_budget < 1.0f;
        int agree = 0, positions = 0, readmis = 0;
        uint64_t srows = 0;
        const uint16_t *ev = eval_tokens(m);
        llm_reset(s);                                      // stage split over positions 0..ctx-1
        double nll = 0.0;
        float nllf = 0.0f;
        int cnt = 0;
        uint32_t ad = 2166136261u;
        for (int k = 0; k + m->ctx <= r->eval; k += m->ctx) {
            for (int i = 0; i < m->ctx; i++) {
                const float *lg = llm_forward(s, ev[k + i], i);
                int am = llm_argmax(lg, m->vout);
                ad = fnv(ad, &am, sizeof am);
                if (r->argmax_out) r->argmax_out[k + i] = am;
                if (r->logits_out && k == 0) memcpy(r->logits_out + (size_t)i * m->vout, lg, sizeof(float) * (size_t)m->vout);
                positions++;
                if (screen) {                              // same hidden state, two-stage head
                    int rows = 0;
                    const float *l2 = llm_rehead(s, r->head_budget, &rows);
                    srows += (uint64_t)rows;
                    agree += llm_argmax(l2, m->vout) == am;
                    for (int c = 0; c < m->clusters; c++)   // rows it read must be bit-identical
                        if (s->mark[c])
                            for (uint32_t q = m->head_off[c]; q < m->head_off[c + 1]; q++) {
                                uint32_t t = m->head_perm[q];
                                readmis += memcmp(&l2[t], &lg[t], sizeof(float)) != 0;
                            }
                }
                if (i + 1 < m->ctx) {
                    float mx = lg[am], sum = 0.0f;
                    for (int v = 0; v < m->vout; v++) sum += f_exp(lg[v] - mx);
                    float l = llm_log(sum) - (lg[ev[k + i + 1]] - mx);
                    nllf += l;
                    nll += l;
                    cnt++;
                }
            }
        }
        fu_t b = {.f = nllf};
        out("ENG {\"label\":\"%s\",\"test\":\"eval\",\"targets\":%d,\"nll\":%.6f,\"nll_sum_bits\":\"%08x\","
            "\"argmax_digest\":\"%08x\"", r->label, cnt, nll / cnt, b.u, ad);
        if (screen)
            out(",\"screen_budget\":%.3f,\"screen_rows\":%.1f,\"screen_top1_agree\":%.4f,\"screen_read_mismatch\":%d",
                r->head_budget, (double)srows / positions, (double)agree / positions, readmis);
        if (s->p.cycles) print_stages(s, out, cpu, positions);
        out("}\n");
        llm_set_head_budget(s, r->head_budget);
    }
}
