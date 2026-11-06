// ESP32-S3 runner for the engine (engine/DESIGN.md).
//
// The model pack (engine/export.py) is flashed to the "model" partition. At
// boot the int4 tensors are copied to SRAM (up to SRAM_BUDGET) or PSRAM; the
// PLE table, RoPE tables and token strings stay in flash (row lookups only).
// Then the shared test runner prints "ENG {json}" lines for three set-ups:
//   pie2  PIE kernels, both cores     (the fast path; also timing repeats)
//   pie1  PIE kernels, one core
//   ref1  scalar C kernels, one core  (the host runs these same kernels)
// All three must print the same digests as each other and as the PC.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "llm.h"

#define HEAD_BUDGET 0.15f          // two-stage head: share of output rows read (engine/test.py passes the same to the PC)
#define SRAM_RESERVE (32 * 1024)   // internal RAM left for stacks, printf and the RTOS

void pie_gemv_q4g(const uint8_t *W, const int8_t *x, int32_t *y, int rows, int groups, int bpg);
void pie_gemv_s8s(const int8_t *W, const int8_t *x, int32_t *y, int rows, int n16, int stride);

static size_t g_sram_used, g_psram_used, g_flash_used;

static void *alloc(size_t n, int fast) {
    void *p = NULL;
    if (fast) p = heap_caps_aligned_alloc(64, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
    if (!p) p = heap_caps_aligned_alloc(64, n, MALLOC_CAP_SPIRAM);
    if (p) memset(p, 0, n);
    return p;
}

// Where each tensor lives. Streamed every token -> SRAM while the budget lasts,
// else PSRAM. Row lookups (PLE table) and per-position tables (RoPE) stay in flash.
static const void *in_place(const char *name, const void *src, size_t n, int kind) {
    (void)name; (void)n; (void)kind;
    return src;
}

// Called after llm_new, so the engine's own buffers already sit in SRAM and
// weights only take what is left.
static int g_has_head;   // model carries a clustered head copy: "emb" is then only looked up by row
static int g_skip_down;  // ReLU model with downT: the dense down codes are never read

static const void *place(const char *name, const void *src, size_t n, int kind) {
    size_t ln = strlen(name);
    if (!strcmp(name, "ple_table") || !strncmp(name, "rope_", 5) || (g_has_head && !strcmp(name, "emb")) ||
        (g_skip_down && kind == 0 && ln > 5 && !strcmp(name + ln - 5, ".down"))) {
        g_flash_used += n;
        return src;
    }
    void *p = NULL;
    if ((kind == 2 && n <= 4096) || (heap_caps_get_free_size(MALLOC_CAP_INTERNAL) >= n + SRAM_RESERVE &&
                      heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL) >= n + 64)) {
        p = heap_caps_aligned_alloc(64, n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
        if (p) g_sram_used += n;
    }
    if (!p) {
        p = heap_caps_aligned_alloc(64, n, MALLOC_CAP_SPIRAM);
        if (!p) { printf("ENG {\"test\":\"error\",\"what\":\"alloc %s\"}\n", name); abort(); }
        g_psram_used += n;
    }
    memcpy(p, src, n);
    return p;
}

// ---- two cores: core 1 spins on a sequence number, no RTOS wake-up per call
static volatile par_body_fn g_fn;
static void *volatile g_ctx;
static volatile int g_lo, g_hi;
static volatile uint32_t g_seq, g_done;

static void worker(void *arg) {
    uint32_t seen = 0;
    for (;;) {
        while (g_seq == seen) { }
        seen = g_seq;
        __asm__ volatile("memw");
        g_fn(g_ctx, g_lo, g_hi, 1);
        __asm__ volatile("memw");
        g_done = seen;
    }
}

static void par2(par_body_fn fn, void *ctx, int n) {
    int mid = n / 2;                     // same split as the host's --split
    g_fn = fn; g_ctx = ctx; g_lo = mid; g_hi = n;
    __asm__ volatile("memw");
    uint32_t seq = g_seq + 1;
    g_seq = seq;
    fn(ctx, 0, mid, 0);
    while (g_done != seq) { }
    __asm__ volatile("memw");
}

static uint32_t cycles(void) { return esp_cpu_get_cycle_count(); }

static int out(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vprintf(fmt, ap);
    va_end(ap);
    return r;
}

void pie_colsum_q4(const uint8_t *const *rows, const int8_t *z, int n, int off, int half, uint8_t *raw);

// QACC as pie_colsum_q4 stores it: lanes 0-7 in bytes 0-19, lanes 8-15 in bytes
// 32-51; 20-bit two's complement fields, little-endian (checked by the unit test).
// Aligned word loads and constant shifts only (a memcpy per lane cost ~1000
// cycles per call, 4x the kernel itself).
#define LANE(w, i) ((int32_t)((((i) * 20 % 32 <= 12 ? (w)[(i) * 20 / 32] >> ((i) * 20 % 32)                       \
                                 : ((w)[(i) * 20 / 32] >> ((i) * 20 % 32)) | ((w)[(i) * 20 / 32 + 1] << ((32 - (i) * 20 % 32) & 31))) \
                                & 0xFFFFFu) << 12) >> 12)
static inline void unpack_qacc(const uint8_t *raw, int32_t *lane) {
    const uint32_t *a = (const uint32_t *)raw, *b = (const uint32_t *)(raw + 32);
    lane[0] = LANE(a, 0);  lane[1] = LANE(a, 1);  lane[2] = LANE(a, 2);  lane[3] = LANE(a, 3);
    lane[4] = LANE(a, 4);  lane[5] = LANE(a, 5);  lane[6] = LANE(a, 6);  lane[7] = LANE(a, 7);
    lane[8] = LANE(b, 0);  lane[9] = LANE(b, 1);  lane[10] = LANE(b, 2); lane[11] = LANE(b, 3);
    lane[12] = LANE(b, 4); lane[13] = LANE(b, 5); lane[14] = LANE(b, 6); lane[15] = LANE(b, 7);
}
#undef LANE

static void colsum_pie(const uint8_t *const *rows, const int8_t *z, int n, int off, int half, int32_t *out) {
    uint8_t raw[64] __attribute__((aligned(16)));
    pie_colsum_q4(rows, z, n, off, half, raw);
    unpack_qacc(raw, out);
}

// Unit test of pie_colsum_q4 against a scalar reference, before the engine uses it.
static void test_colsum(void) {
    enum { N = 64, RB = 64 };
    uint8_t *rows = heap_caps_aligned_alloc(16, N * RB, MALLOC_CAP_INTERNAL);
    const uint8_t **ptr = malloc(N * sizeof *ptr);
    int8_t *z = malloc(N);
    uint8_t *raw = heap_caps_aligned_alloc(16, 64, MALLOC_CAP_INTERNAL);
    uint32_t r = 12345;
    for (int i = 0; i < N * RB; i++) { r = r * 1103515245u + 12345u; rows[i] = (uint8_t)(r >> 16); }
    for (int k = 0; k < N; k++) {
        r = r * 1103515245u + 12345u;
        z[k] = (int8_t)((int)((r >> 16) % 255) - 127);
        ptr[k] = rows + (size_t)((k * 37) % N) * RB;          // shuffled row order, like a live list
    }
    int bad = 0, cases = 0, first = -1;
    int32_t lane[16], ref[16];
    for (int n = 0; n <= N; n += 13) {
        for (int off = 0; off < RB; off += 16)
            for (int half = 0; half < 2; half++) {
                pie_colsum_q4(ptr, z, n, off, half, raw);
                unpack_qacc(raw, lane);
                for (int l = 0; l < 16; l++) {
                    ref[l] = 0;
                    for (int k = 0; k < n; k++) {
                        uint8_t b = ptr[k][off + l];
                        ref[l] += z[k] * ((int)(half ? b >> 4 : b & 15) - 8);
                    }
                    if (ref[l] != lane[l]) { bad++; if (first < 0) first = l; }
                }
                cases++;
            }
    }
    // cost per call, 13 live rows (a typical group): kernel alone, kernel + unpack, rows in SRAM / cold PSRAM
    uint8_t *prow = heap_caps_aligned_alloc(64, 1 << 18, MALLOC_CAP_SPIRAM);
    const uint8_t **pp = malloc(13 * sizeof *pp);
    uint32_t c_k = UINT32_MAX, c_ku = UINT32_MAX, c_ps = UINT32_MAX;
    for (int rep = 0; rep < 5; rep++) {
        uint32_t t = cycles();
        for (int i = 0; i < 100; i++) pie_colsum_q4(ptr, z, 13, 16, 1, raw);
        t = cycles() - t; if (t < c_k) c_k = t;
        t = cycles();
        for (int i = 0; i < 100; i++) { pie_colsum_q4(ptr, z, 13, 16, 1, raw); unpack_qacc(raw, lane); }
        t = cycles() - t; if (t < c_ku) c_ku = t;
        t = cycles();
        for (int i = 0; i < 100; i++) {                   // fresh PSRAM lines every call
            for (int k = 0; k < 13; k++) pp[k] = prow + ((size_t)(i * 13 + k) * 64 * 3 + rep * 97 * 64) % ((1 << 18) - 64);
            pie_colsum_q4(pp, z, 13, 0, 0, raw);
        }
        t = cycles() - t; if (t < c_ps) c_ps = t;
    }
    heap_caps_free(prow); free(pp);
    printf("ENG {\"test\":\"unit_colsum\",\"cases\":%d,\"bad_lanes\":%d,\"first_bad_lane\":%d,"
           "\"cyc_call13_sram\":%.0f,\"cyc_call13_unpack\":%.0f,\"cyc_call13_cold_psram\":%.0f}\n",
           cases, bad, first, c_k / 100.0, c_ku / 100.0, c_ps / 100.0);
    if (bad) {
        pie_colsum_q4(ptr, z, 5, 0, 0, raw);
        unpack_qacc(raw, lane);
        printf("ENG {\"test\":\"unit_colsum_dump\",\"raw\":\"");
        for (int i = 0; i < 64; i++) printf("%02x", raw[i]);
        printf("\",\"ref\":[");
        for (int l = 0; l < 16; l++) {
            int32_t a = 0;
            for (int k = 0; k < 5; k++) a += z[k] * ((int)(ptr[k][l] & 15) - 8);
            printf("%s%d", l ? "," : "", (int)a);
        }
        printf("]}\n");
    }
    heap_caps_free(rows); free(ptr); free(z); heap_caps_free(raw);
}

// Costs of the small pieces around the matvecs, to find where non-streaming
// time goes (attention, PLE). Cycles per call, best of 3.
static volatile float g_fsink;
static void noop_body(void *c, int lo, int hi, int w) { (void)c; (void)lo; (void)hi; (void)w; }

static void microbench(void) {
    enum { N = 1000 };
    uint32_t best[6] = {UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX, UINT32_MAX};
    uint8_t *ps = heap_caps_aligned_alloc(64, 1 << 20, MALLOC_CAP_SPIRAM);
    uint8_t *sr = heap_caps_aligned_alloc(64, 4096, MALLOC_CAP_INTERNAL);
    if (!ps || !sr) { printf("ENG {\"test\":\"micro\",\"error\":\"alloc\"}\n"); return; }
    int8_t *x = heap_caps_aligned_alloc(64, 128, MALLOC_CAP_INTERNAL);
    int32_t *y = heap_caps_aligned_alloc(64, 64 * 2 * 4, MALLOC_CAP_INTERNAL);
    memset(ps, 0x5A, 1 << 20); memset(sr, 0x5A, 4096); memset(x, 3, 128);
    for (int rep = 0; rep < 3; rep++) {
        uint32_t t = cycles(); float a = 0;
        for (int i = 0; i < N; i++) a += llm_exp(-0.01f * (float)i);
        g_fsink = a; t = cycles() - t; if (t < best[0]) best[0] = t;
        t = cycles(); a = 0;
        for (int i = 0; i < N; i++) a += llm_recip(1.0f + (float)i);
        g_fsink = a; t = cycles() - t; if (t < best[1]) best[1] = t;
        t = cycles();
        for (int i = 0; i < N; i++) par2(noop_body, NULL, 2);
        t = cycles() - t; if (t < best[2]) best[2] = t;
        t = cycles();
        for (int i = 0; i < N; i++) pie_gemv_q4g(sr, x, y, 64, 2, 2);               // 4 KB, SRAM
        t = cycles() - t; if (t < best[3]) best[3] = t;
        t = cycles();
        for (int i = 0; i < 200; i++) pie_gemv_q4g(ps + (size_t)i * 4096, x, y, 64, 2, 2);  // 4 KB, cold PSRAM
        t = (cycles() - t) * (N / 200); if (t < best[4]) best[4] = t;
        t = cycles(); a = 0;
        for (int i = 0; i < N; i++) a += *(volatile float *)(ps + (size_t)((i * 4099) & 0xFFFFC));  // random PSRAM word
        g_fsink = a; t = cycles() - t; if (t < best[5]) best[5] = t;
    }
    printf("ENG {\"test\":\"micro\",\"cyc_exp\":%.1f,\"cyc_recip\":%.1f,\"cyc_par2_empty\":%.1f,"
           "\"cyc_q4g_4KB_sram\":%.1f,\"cyc_q4g_4KB_psram_cold\":%.1f,\"cyc_psram_random_load\":%.1f}\n",
           best[0] / (double)N, best[1] / (double)N, best[2] / (double)N, best[3] / (double)N,
           best[4] / (double)N, best[5] / (double)N);
    heap_caps_free(ps); heap_caps_free(sr); heap_caps_free(x); heap_caps_free(y);
}

void demo_loop(llm_t *s, const llm_model_t *m, float head_budget, int psram_mhz);

void app_main(void) {
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) { printf("ENG {\"test\":\"error\",\"what\":\"no model partition\"}\n"); return; }
    uint32_t hdr[16];
    esp_partition_read(part, 0, hdr, sizeof hdr);
    size_t total = hdr[14];                               // magic(2 words) + 13 fields; total is last
    if ((memcmp(hdr, "MCULM01", 8) && memcmp(hdr, "MCULM03", 8)) || total > part->size) {
        printf("ENG {\"test\":\"error\",\"what\":\"bad pack header\"}\n");
        return;
    }
    const void *blob;
    esp_partition_mmap_handle_t mh;
    ESP_ERROR_CHECK(esp_partition_mmap(part, 0, (total + 0xFFFF) & ~0xFFFFu, ESP_PARTITION_MMAP_DATA, &blob, &mh));

    llm_platform_t pie2 = {pie_gemv_q4g, pie_gemv_s8s, colsum_pie, par2, cycles, in_place};
    static llm_model_t m;                                  // ~3 KB: not on the 3.5 KB default main stack
    int e = llm_load(&m, blob, total, &pie2);            // shapes only, nothing copied
    if (e) { printf("ENG {\"test\":\"error\",\"what\":\"load %d\"}\n", e); return; }
    llm_t *s = llm_new(&m, &pie2, alloc);                  // engine buffers get SRAM first
    printf("ENG {\"test\":\"heapcheck\",\"after\":\"llm_new\",\"ok\":%d}\n", heap_caps_check_integrity_all(true));
    if (!s) { printf("ENG {\"test\":\"error\",\"what\":\"llm_new\"}\n"); return; }
    xTaskCreatePinnedToCore(worker, "par", 4096, NULL, configMAX_PRIORITIES - 1, NULL, 1);
#ifndef LLM_DEMO
    test_colsum();
    microbench();                                          // before weights take the SRAM
#endif
    printf("ENG {\"test\":\"heapcheck\",\"after\":\"microbench\",\"ok\":%d}\n", heap_caps_check_integrity_all(true));
    g_has_head = m.clusters > 0 && !m.emb_row;          // separate head copy: emb is lookup-only
    g_skip_down = m.act != ACT_SILU && m.L[0].downT != NULL;
    pie2.place = place;
    uint32_t t0 = cycles();
    llm_load(&m, blob, total, &pie2);                      // now copy weights into what is left
    uint32_t t1 = cycles();
    llm_set_platform(s, &pie2);
#ifdef LLM_DEMO
    demo_loop(s, &m, HEAD_BUDGET, CONFIG_SPIRAM_SPEED);   // never returns
#endif

    printf("ENG {\"label\":\"board\",\"test\":\"config\",\"vocab\":%d,\"d\":%d,\"layers\":%d,\"heads\":%d,"
           "\"ffn\":%d,\"ple\":%d,\"act\":%d,\"ctx\":%d,\"pack_bytes\":%u,\"load_ms\":%.1f,"
           "\"weights_sram\":%u,\"weights_psram\":%u,\"flash_inplace\":%u,\"sram_free\":%u,\"psram_free\":%u}\n",
           m.vocab, m.d, m.layers, m.heads, m.ffn, m.ple, m.act, m.ctx, (unsigned)total,
           (double)(t1 - t0) / 240000.0, (unsigned)g_sram_used, (unsigned)g_psram_used, (unsigned)g_flash_used,
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));

    llm_run_t r2 = {"pie2", m.n_prompts, m.n_gen, m.n_eval, 5, HEAD_BUDGET, 64, NULL, NULL, 1, 1};
    llm_run_tests(s, &m, &r2, out, 240.0);
    // same fast path with the full output head: the like-for-like number against engines without a two-stage head
    llm_run_t rf = {"pie2full", 0, m.n_gen, 0, 3, 1.0f, 0, NULL, NULL, 0, 0};
    llm_run_tests(s, &m, &rf, out, 240.0);
    // experiment: KV cache pre-touched before each attention (docs/phase3-engine.md, attention)
    llm_set_debug(s, LLM_DBG_WARM_KV);
    llm_run_t rw = {"pie2warm", 0, m.n_gen, m.n_eval, 3, HEAD_BUDGET, 0, NULL, NULL, 0, 0};
    llm_run_tests(s, &m, &rw, out, 240.0);
    llm_set_debug(s, 0);

    llm_platform_t pie1 = pie2;
    pie1.par = NULL;
    llm_set_platform(s, &pie1);
    llm_run_t r1 = {"pie1", m.n_prompts, m.n_gen, m.n_eval, 2, HEAD_BUDGET, 64, NULL, NULL, 1, 0};
    llm_run_tests(s, &m, &r1, out, 240.0);

    llm_platform_t ref1 = pie1;
    ref1.q4g = llm_ref_q4g;
    ref1.s8 = llm_ref_s8;
    ref1.colsum = llm_ref_colsum;
    llm_set_platform(s, &ref1);
    llm_run_t r0 = {"ref1", m.n_prompts, m.n_gen, m.n_eval, 0, HEAD_BUDGET, 64, NULL, NULL, 1, 0};
    llm_run_tests(s, &m, &r0, out, 240.0);

    printf("ENG {\"test\":\"done\"}\n");
}
