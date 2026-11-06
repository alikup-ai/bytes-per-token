// Roofline bench for ESP32-S3 N16R8 (plan phase 1). Every result is one line
//   BENCH {"test": ..., ...}
// on the console; bench/parse.py turns a captured log into roofline.json.
// Each measurement is best-of-REPS (least interrupted), as in esp32-tinyLLM's
// bandwidth_bench.
#include <stdio.h>
#include <string.h>
#include <inttypes.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_cpu.h"
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "esp_cache.h"
#include "esp_async_memcpy.h"

#define REPS 5
#define F_CPU 240e6

void pie_gemv_s8(const int8_t *W, const int8_t *x, int32_t *y, int rows, int n16);
void pie_gemv_q4(const uint8_t *W, const int8_t *x, int32_t *y, int rows, int n32);

static volatile uint32_t g_sink;
static inline uint32_t cc(void) { return esp_cpu_get_cycle_count(); }

static void *alloc(size_t n, uint32_t caps) {
    void *p = heap_caps_aligned_alloc(64, n, caps);
    if (!p) { printf("BENCH {\"test\":\"alloc_fail\",\"bytes\":%u}\n", (unsigned)n); abort(); }
    return p;
}

static uint32_t rng = 0x12345678;
static inline uint32_t xs(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static void fill(void *p, size_t n) { uint8_t *b = p; for (size_t i = 0; i < n; i++) b[i] = xs(); }

// ---- 1. sequential read bandwidth ---------------------------------------
static uint32_t read_words(const volatile uint32_t *p, size_t n) {
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (size_t i = 0; i + 3 < n; i += 4) { a0 += p[i]; a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3]; }
    return a0 + a1 + a2 + a3;
}

static double seq_MBps(const void *p, size_t bytes) {
    uint32_t best = UINT32_MAX;
    for (int r = 0; r < REPS; r++) {
        uint32_t t0 = cc();
        g_sink = read_words(p, bytes / 4);
        uint32_t dt = cc() - t0;
        if (dt < best) best = dt;
    }
    return bytes / (best / F_CPU) / 1e6;
}

// Dual-core: the worker on core 1 reads the second half while core 0 reads
// the first; wall time from esp_timer covers both.
typedef struct { const void *p; size_t bytes; SemaphoreHandle_t go, done; int kind; } job_t;
static job_t g_job;
static int8_t *g_x;
static int32_t *g_y1;

static void run_job(const job_t *j) {
    if (j->kind == 0) g_sink = read_words(j->p, j->bytes / 4);
    else if (j->kind == 1) pie_gemv_s8(j->p, g_x, g_y1, j->bytes / 96, 6);
    else pie_gemv_q4(j->p, g_x, g_y1, j->bytes / 48, 3);
}

static void worker(void *arg) {
    for (;;) {
        xSemaphoreTake(g_job.go, portMAX_DELAY);
        run_job(&g_job);
        xSemaphoreGive(g_job.done);
    }
}

static double dual_MBps(const uint8_t *p, size_t bytes, int kind) {
    int64_t best = INT64_MAX;
    size_t half = bytes / 2;
    for (int r = 0; r < REPS; r++) {
        job_t mine = { .p = p, .bytes = half, .kind = kind };
        g_job.p = p + half; g_job.bytes = half; g_job.kind = kind;
        int64_t t0 = esp_timer_get_time();
        xSemaphoreGive(g_job.go);
        run_job(&mine);
        xSemaphoreTake(g_job.done, portMAX_DELAY);
        int64_t dt = esp_timer_get_time() - t0;
        if (dt < best) best = dt;
    }
    return (double)bytes / best;   // bytes/us == MB/s
}

// ---- 2. flash random row latency ----------------------------------------
static double flash_row_us(const uint8_t *base, size_t span, size_t row, int n) {
    uint32_t acc = 0, t0 = cc();
    for (int i = 0; i < n; i++) {
        size_t off = (xs() % (span - row)) & ~(size_t)3;
        const volatile uint32_t *q = (const volatile uint32_t *)(base + off);
        for (size_t b = 0; b < row / 4; b++) acc += q[b];
    }
    uint32_t dt = cc() - t0;
    g_sink = acc;
    return dt / F_CPU * 1e6 / n;
}

// ---- 3. gemv kernels: scalar references + cycles/MAC ----------------------
static void ref_s8(const int8_t *W, const int8_t *x, int32_t *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const int8_t *w = W + (size_t)r * cols;
        int32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;   // 4 chains, as esp32-tinyLLM found necessary
        for (int j = 0; j < cols; j += 4) {
            a0 += w[j] * x[j]; a1 += w[j + 1] * x[j + 1]; a2 += w[j + 2] * x[j + 2]; a3 += w[j + 3] * x[j + 3];
        }
        y[r] = a0 + a1 + a2 + a3;
    }
}

static void ref_q4(const uint8_t *W, const int8_t *x, int32_t *y, int rows, int cols) {
    int rb = cols / 2;
    for (int r = 0; r < rows; r++) {
        const uint8_t *w = W + (size_t)r * rb;
        int32_t acc = 0;
        for (int b = 0; b < cols / 32; b++)
            for (int j = 0; j < 16; j++) {
                uint8_t byte = w[b * 16 + j];
                acc += ((byte & 15) - 8) * x[b * 32 + j] + ((byte >> 4) - 8) * x[b * 32 + 16 + j];
            }
        y[r] = acc;
    }
}

typedef enum { K_REF_S8, K_PIE_S8, K_REF_Q4, K_PIE_Q4 } kern_t;
static const char *KN[] = { "scalar_s8", "pie_s8", "scalar_q4", "pie_q4" };

static double gemv_cpm(kern_t k, const void *W, int rows, int cols, int32_t *y) {
    uint32_t best = UINT32_MAX;
    for (int r = 0; r < REPS; r++) {
        uint32_t t0 = cc();
        switch (k) {
        case K_REF_S8: ref_s8(W, g_x, y, rows, cols); break;
        case K_PIE_S8: pie_gemv_s8(W, g_x, y, rows, cols / 16); break;
        case K_REF_Q4: ref_q4(W, g_x, y, rows, cols); break;
        case K_PIE_Q4: pie_gemv_q4(W, g_x, y, rows, cols / 32); break;
        }
        uint32_t dt = cc() - t0;
        if (dt < best) best = dt;
    }
    return (double)best / ((double)rows * cols);
}

static void gemv_suite(const char *tier, const void *W, int rows, int cols, int q4) {
    if (rows > 16384) rows = 16384;   // 1.5 MB of s8 weights: still 24x the data cache
    // Outputs in PSRAM: 4 bytes per row against 48-96 bytes of weights read.
    int32_t *ya = alloc(rows * 4, MALLOC_CAP_SPIRAM), *yb = alloc(rows * 4, MALLOC_CAP_SPIRAM);
    kern_t ref = q4 ? K_REF_Q4 : K_REF_S8, pie = q4 ? K_PIE_Q4 : K_PIE_S8;
    double c_ref = gemv_cpm(ref, W, rows, cols, ya);
    double c_pie = gemv_cpm(pie, W, rows, cols, yb);
    int ok = memcmp(ya, yb, rows * 4) == 0;
    double wbytes = (double)rows * cols * (q4 ? 0.5 : 1.0);
    for (int i = 0; i < 2; i++) {
        double cpm = i ? c_pie : c_ref;
        double s = cpm * rows * cols / F_CPU;
        printf("BENCH {\"test\":\"gemv\",\"kernel\":\"%s\",\"tier\":\"%s\",\"rows\":%d,\"cols\":%d,"
               "\"cyc_per_mac\":%.3f,\"weight_MBps\":%.1f,\"match_ref\":%d}\n",
               KN[i ? pie : ref], tier, rows, cols, cpm, wbytes / s / 1e6, ok);
    }
    heap_caps_free(ya); heap_caps_free(yb);
}

// ---- 5. DMA PSRAM->SRAM concurrently with compute --------------------------
static SemaphoreHandle_t dma_sem;
static bool IRAM_ATTR dma_done(async_memcpy_handle_t h, async_memcpy_event_t *e, void *arg) {
    BaseType_t hp = pdFALSE;
    xSemaphoreGiveFromISR(dma_sem, &hp);
    return hp == pdTRUE;
}

#define CHUNK (32 * 1024)
typedef struct { async_memcpy_handle_t h; uint8_t *src; uint8_t *dst[2]; size_t total; int64_t us; int inflight; SemaphoreHandle_t go, done; } dma_job_t;
static dma_job_t g_dma;

// inflight 1: issue a chunk, wait for it, issue the next (the original method: the
// interrupt + semaphore gap between chunks is counted as DMA time).
// inflight 2: keep two chunks queued, so the engine always has the next one.
static int64_t dma_stream(dma_job_t *d) {
    size_t n = d->total / CHUNK, issued = 0, done = 0;
    int64_t t0 = esp_timer_get_time();
    while (done < n) {
        while (issued < n && issued - done < (size_t)d->inflight) {
            ESP_ERROR_CHECK(esp_async_memcpy(d->h, d->dst[issued & 1], d->src + issued * CHUNK, CHUNK, dma_done, NULL));
            issued++;
        }
        xSemaphoreTake(dma_sem, portMAX_DELAY);
        done++;
    }
    return esp_timer_get_time() - t0;
}

static void dma_worker(void *arg) {
    for (;;) {
        xSemaphoreTake(g_dma.go, portMAX_DELAY);
        g_dma.us = dma_stream(&g_dma);
        xSemaphoreGive(g_dma.done);
    }
}

// burst: GDMA burst size in bytes. The IDF default (16) may starve PSRAM, so
// the suite runs once per burst size.
static void dma_suite(uint8_t *psram_src, size_t total, const int8_t *Wsram, int rows, int burst, int inflight) {
    async_memcpy_config_t cfg = ASYNC_MEMCPY_DEFAULT_CONFIG();
    cfg.dma_burst_size = burst;
    esp_err_t err = esp_async_memcpy_install(&cfg, &g_dma.h);
    if (err != ESP_OK) { printf("BENCH {\"test\":\"dma\",\"burst\":%d,\"error\":%d}\n", burst, err); return; }
    if (!dma_sem) {
        dma_sem = xSemaphoreCreateCounting(2, 0);
        g_dma.dst[0] = alloc(CHUNK, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
        g_dma.dst[1] = alloc(CHUNK, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
        g_dma.go = xSemaphoreCreateBinary(); g_dma.done = xSemaphoreCreateBinary();
        xTaskCreatePinnedToCore(dma_worker, "dma", 4096, NULL, 5, NULL, 1);
    }
    g_dma.src = psram_src; g_dma.total = total; g_dma.inflight = inflight;
    esp_cache_msync(psram_src, total, ESP_CACHE_MSYNC_FLAG_DIR_C2M);

    // (a) DMA alone
    xSemaphoreGive(g_dma.go); xSemaphoreTake(g_dma.done, portMAX_DELAY);
    int64_t dma_alone = g_dma.us;
    // (b) compute alone: PIE gemv over SRAM weights, repeated to last ~as long
    int32_t *y = alloc(rows * 4, MALLOC_CAP_INTERNAL);
    int reps = 1;
    int64_t t0 = esp_timer_get_time();
    pie_gemv_s8(Wsram, g_x, y, rows, 6);
    int64_t one = esp_timer_get_time() - t0;
    if (one > 0) reps = (int)(dma_alone / one) + 1;
    t0 = esp_timer_get_time();
    for (int i = 0; i < reps; i++) pie_gemv_s8(Wsram, g_x, y, rows, 6);
    int64_t comp_alone = esp_timer_get_time() - t0;
    // (c) both at once
    xSemaphoreGive(g_dma.go);
    t0 = esp_timer_get_time();
    for (int i = 0; i < reps; i++) pie_gemv_s8(Wsram, g_x, y, rows, 6);
    int64_t comp_with = esp_timer_get_time() - t0;
    xSemaphoreTake(g_dma.done, portMAX_DELAY);
    int64_t dma_with = g_dma.us;
    printf("BENCH {\"test\":\"dma_overlap\",\"inflight\":%d,\"burst\":%d,\"bytes\":%u,\"dma_alone_MBps\":%.1f,\"dma_with_compute_MBps\":%.1f,"
           "\"compute_alone_us\":%" PRId64 ",\"compute_with_dma_us\":%" PRId64 "}\n",
           inflight, burst, (unsigned)total, (double)total / dma_alone, (double)total / dma_with, comp_alone, comp_with);
    heap_caps_free(y);
    esp_async_memcpy_uninstall(g_dma.h);
}

// ---- 6. PSRAM bus shared by CPU reads and GDMA -----------------------------
// Can GDMA add bandwidth on top of the cores' own reads? Agents: core 0 reads
// region 0, core 1 reads region 1, GDMA channels 0/1 copy regions 2/3 into
// SRAM (32 KB chunks, burst 64). Each agent times itself; the aggregate is all
// bytes over the slowest agent.
#define BUS_REGION (1u << 20)
static SemaphoreHandle_t bus_sem;
static bool IRAM_ATTR bus_done(async_memcpy_handle_t h, async_memcpy_event_t *e, void *arg) {
    BaseType_t hp = pdFALSE;
    xSemaphoreGiveFromISR(bus_sem, &hp);
    return hp == pdTRUE;
}
typedef struct { async_memcpy_handle_t h[2]; uint8_t *dst[2]; const uint8_t *src; int nch; int64_t us; SemaphoreHandle_t go, done; } bus_dma_t;
static bus_dma_t g_bus;

static void bus_dma_task(void *arg) {
    for (;;) {
        xSemaphoreTake(g_bus.go, portMAX_DELAY);
        int64_t t0 = esp_timer_get_time();
        for (size_t off = 0; off < BUS_REGION; off += CHUNK) {
            for (int c = 0; c < g_bus.nch; c++)
                ESP_ERROR_CHECK(esp_async_memcpy(g_bus.h[c], g_bus.dst[c], (void *)(g_bus.src + (2 + c) * BUS_REGION + off), CHUNK, bus_done, NULL));
            for (int c = 0; c < g_bus.nch; c++) xSemaphoreTake(bus_sem, portMAX_DELAY);
        }
        g_bus.us = esp_timer_get_time() - t0;
        xSemaphoreGive(g_bus.done);
    }
}

static void bus_case(const uint8_t *ps, int cpu0, int cpu1, int nch) {
    double best = 0, r0 = 0, r1 = 0, rd = 0;
    for (int rep = 0; rep < REPS; rep++) {
        int64_t t0 = esp_timer_get_time(), c0 = 0, c1 = 0, dm = 0;
        g_bus.src = ps; g_bus.nch = nch;
        if (nch) xSemaphoreGive(g_bus.go);
        if (cpu1) { g_job.p = ps + BUS_REGION; g_job.bytes = BUS_REGION; g_job.kind = 0; xSemaphoreGive(g_job.go); }
        if (cpu0) { g_sink = read_words((const void *)ps, BUS_REGION / 4); c0 = esp_timer_get_time() - t0; }
        if (cpu1) { xSemaphoreTake(g_job.done, portMAX_DELAY); c1 = esp_timer_get_time() - t0; }
        if (nch) { xSemaphoreTake(g_bus.done, portMAX_DELAY); dm = g_bus.us; }
        int64_t mx = c0 > c1 ? c0 : c1;
        if (dm > mx) mx = dm;
        double agg = (double)(cpu0 + cpu1 + nch) * BUS_REGION / mx;
        if (agg > best) {
            best = agg;
            r0 = c0 ? (double)BUS_REGION / c0 : 0;
            r1 = c1 ? (double)BUS_REGION / c1 : 0;
            rd = dm ? (double)nch * BUS_REGION / dm : 0;
        }
    }
    printf("BENCH {\"test\":\"bus_share\",\"cpu0\":%d,\"cpu1\":%d,\"dma_ch\":%d,\"cpu0_MBps\":%.1f,\"cpu1_MBps\":%.1f,"
           "\"dma_MBps\":%.1f,\"aggregate_MBps\":%.1f}\n", cpu0, cpu1, nch, r0, r1, rd, best);
}

static void bus_suite(uint8_t *ps) {
    async_memcpy_config_t cfg = ASYNC_MEMCPY_DEFAULT_CONFIG();
    cfg.dma_burst_size = 64;
    for (int c = 0; c < 2; c++) {
        esp_err_t err = esp_async_memcpy_install(&cfg, &g_bus.h[c]);
        if (err != ESP_OK) { printf("BENCH {\"test\":\"bus_share\",\"error\":%d,\"ch\":%d}\n", err, c); return; }
        g_bus.dst[c] = alloc(CHUNK, MALLOC_CAP_INTERNAL | MALLOC_CAP_DMA);
    }
    bus_sem = xSemaphoreCreateCounting(4, 0);
    g_bus.go = xSemaphoreCreateBinary(); g_bus.done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(bus_dma_task, "busdma", 4096, NULL, 7, NULL, 1);   // above the core-1 reader
    esp_cache_msync(ps, 4 * BUS_REGION, ESP_CACHE_MSYNC_FLAG_DIR_C2M);
    bus_case(ps, 1, 0, 0);
    bus_case(ps, 1, 1, 0);
    bus_case(ps, 0, 0, 1);
    bus_case(ps, 0, 0, 2);
    bus_case(ps, 1, 0, 1);
    bus_case(ps, 1, 1, 1);
    bus_case(ps, 1, 1, 2);
    for (int c = 0; c < 2; c++) { esp_async_memcpy_uninstall(g_bus.h[c]); heap_caps_free(g_bus.dst[c]); }
}

void app_main(void) {
    vTaskDelay(pdMS_TO_TICKS(500));
    printf("BENCH {\"test\":\"boot\",\"psram_free\":%u,\"sram_free\":%u}\n",
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    g_job.go = xSemaphoreCreateBinary(); g_job.done = xSemaphoreCreateBinary();
    xTaskCreatePinnedToCore(worker, "w1", 4096, NULL, 5, NULL, 1);

    g_x = alloc(512, MALLOC_CAP_INTERNAL); fill(g_x, 512);
    g_y1 = alloc(32768 * 4, MALLOC_CAP_SPIRAM);

    // Tiers. PSRAM buffer is 4 MB, far past the 64 KB data cache.
    const size_t SRAM_B = 96 * 1024, PS_B = 4u << 20;
    uint8_t *sram = alloc(SRAM_B, MALLOC_CAP_INTERNAL);
    uint8_t *ps = alloc(PS_B, MALLOC_CAP_SPIRAM);
    fill(sram, SRAM_B); fill(ps, PS_B);
    bus_suite(ps);

    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "bench");
    const uint8_t *fl = NULL; esp_partition_mmap_handle_t mh;
    size_t FL_B = 4u << 20;
    if (part) {
        ESP_ERROR_CHECK(esp_partition_erase_range(part, 0, FL_B));
        for (size_t off = 0; off < FL_B; off += 64 * 1024)   // real bytes, not erased 0xFF
            ESP_ERROR_CHECK(esp_partition_write(part, off, ps + off, 64 * 1024));
        ESP_ERROR_CHECK(esp_partition_mmap(part, 0, FL_B, ESP_PARTITION_MMAP_DATA, (const void **)&fl, &mh));
    }

    printf("BENCH {\"test\":\"seq_read\",\"tier\":\"sram\",\"cores\":1,\"MBps\":%.1f}\n", seq_MBps(sram, SRAM_B));
    printf("BENCH {\"test\":\"seq_read\",\"tier\":\"psram\",\"cores\":1,\"MBps\":%.1f}\n", seq_MBps(ps, PS_B));
    printf("BENCH {\"test\":\"seq_read\",\"tier\":\"psram\",\"cores\":2,\"MBps\":%.1f}\n", dual_MBps(ps, PS_B, 0));
    if (fl) {
        printf("BENCH {\"test\":\"seq_read\",\"tier\":\"flash\",\"cores\":1,\"MBps\":%.1f}\n", seq_MBps(fl, FL_B));
        printf("BENCH {\"test\":\"seq_read\",\"tier\":\"flash\",\"cores\":2,\"MBps\":%.1f}\n", dual_MBps(fl, FL_B, 0));
        for (int row = 64; row <= 1024; row *= 4)
            printf("BENCH {\"test\":\"flash_row\",\"row_bytes\":%d,\"us\":%.2f}\n", row, flash_row_us(fl, FL_B, row, 2000));
    }

    // gemv: cols 96 = the TinyStories model dim (6 PIE blocks / 3 q4 blocks).
    const int C = 96;
    gemv_suite("sram", sram, SRAM_B / C, C, 0);
    gemv_suite("sram", sram, SRAM_B / (C / 2), C, 1);
    gemv_suite("psram", ps, PS_B / C, C, 0);
    gemv_suite("psram", ps, PS_B / (C / 2), C, 1);
    if (fl) {
        gemv_suite("flash", fl, FL_B / C, C, 0);
        gemv_suite("flash", fl, FL_B / (C / 2), C, 1);
    }
    // 4. PIE on both cores, streaming PSRAM.
    printf("BENCH {\"test\":\"gemv_dual\",\"kernel\":\"pie_s8\",\"tier\":\"psram\",\"weight_MBps\":%.1f}\n", dual_MBps(ps, PS_B, 1));
    printf("BENCH {\"test\":\"gemv_dual\",\"kernel\":\"pie_q4\",\"tier\":\"psram\",\"weight_MBps\":%.1f}\n", dual_MBps(ps, PS_B, 2));

    for (int b = 16; b <= 64; b *= 2) dma_suite(ps, 1u << 20, (const int8_t *)sram, SRAM_B / C, b, 1);
    dma_suite(ps, 1u << 20, (const int8_t *)sram, SRAM_B / C, 64, 2);   // review 2026-09-29: two chunks queued
    printf("BENCH {\"test\":\"done\"}\n");
}
