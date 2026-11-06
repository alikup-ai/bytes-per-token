// Live demo mode (built with -DLLM_DEMO=1, see engine/demo/README.md).
//
// Line protocol on the console UART, 115200 baud:
//   host -> board   I                                   say hello again (READY line)
//                   G max temp_milli topk seed eot n id1 .. idn
//                                                       continue these n prompt tokens
//                   X                                   stop the running generation
//                   H budget                            share of output rows the head reads (1 = full head)
//                   K n id1 .. idn                      add stop tokens for template slots (K 0: clear)
//                   S maxslot temp_milli topk seed eot batch nseg seg..
//                                                       fill a template; seg = 0 n id1 .. idn (fixed text)
//                                                       or 1 (a slot: generate until a stop token, eot or
//                                                       maxslot tokens; the stop token is not emitted)
//                                                       batch 1: fixed text in batches, 0: token by token
//   board -> host   READY {json}                        model shape, clock, head budget
//                   P prefill_ms                        prompt processed (batched)
//                   T id us                             one new token; us = its forward time
//                   E n ms_per_token                    generation finished
//                   Q n ms id1 .. idn                   fixed template tokens processed in ms
//                   R gen forced ms_gen ms_forced       template finished
//                   ERR reason                          command refused (bad syntax or segment, prompt
//                                                       longer than max_prompt, token id outside the vocabulary)
// Tokens stream as they are produced, so the page shows the text growing at
// the real speed of the board.
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "driver/uart.h"
#include "driver/uart_vfs.h"
#include "esp_cpu.h"
#include "llm.h"

#define LINE_MAX_BYTES 4096
#define MAX_PROMPT 240
#define MAX_TPL 480

static uint32_t g_rng = 1;
static inline uint32_t xr(void) { g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5; return g_rng; }

// Greedy when temp <= 0; otherwise sample from the top-k logits at temperature temp.
static int sample(const float *lg, int n, float temp, int topk) {
    if (temp <= 0.0f || topk <= 1) return llm_argmax(lg, n);
    if (topk > 64) topk = 64;
    int idx[64], k = 0;
    float val[64];
    for (int v = 0; v < n; v++) {
        float x = lg[v];
        if (k == topk && x <= val[k - 1]) continue;
        int i = k < topk ? k++ : k - 1;              // insertion into the sorted top-k
        while (i > 0 && val[i - 1] < x) { val[i] = val[i - 1]; idx[i] = idx[i - 1]; i--; }
        val[i] = x;
        idx[i] = v;
    }
    float p[64], sum = 0.0f;
    for (int i = 0; i < k; i++) { p[i] = expf((val[i] - val[0]) / temp); sum += p[i]; }
    float r = (float)(xr() >> 8) * (1.0f / 16777216.0f) * sum;
    for (int i = 0; i < k; i++) { r -= p[i]; if (r <= 0.0f) return idx[i]; }
    return idx[k - 1];
}

static int read_line(char *buf, int cap) {
    int n = 0;
    for (;;) {
        uint8_t c;
        if (uart_read_bytes(UART_NUM_0, &c, 1, portMAX_DELAY) != 1) continue;
        if (c == '\r') continue;
        if (c == '\n') { buf[n] = 0; return n; }
        if (n < cap - 1) buf[n++] = (char)c;
    }
}

static int stop_requested(void) {
    uint8_t c;
    while (uart_read_bytes(UART_NUM_0, &c, 1, 0) == 1)
        if (c == 'X') return 1;
    return 0;
}

static void ready(const llm_model_t *m, float head_budget, int psram_mhz) {
    printf("READY {\"vocab\":%d,\"vout\":%d,\"d\":%d,\"layers\":%d,\"heads\":%d,\"ffn\":%d,\"ple\":%d,"
           "\"ctx\":%d,\"head_budget\":%.2f,\"psram_mhz\":%d,\"max_prompt\":%d}\n",
           m->vocab, m->vout, m->d, m->layers, m->heads, m->ffn, m->ple, m->ctx, head_budget, psram_mhz,
           MAX_PROMPT < m->ctx - 1 ? MAX_PROMPT : m->ctx - 1);
    fflush(stdout);
}

static uint8_t g_stop[32768 / 8];
static inline int is_stop(int t) { return t >= 0 && t < 32768 && (g_stop[t >> 3] >> (t & 7)) & 1; }

static int add_stops(char *p) {
    int n = (int)strtol(p, &p, 10);
    if (n == 0) { memset(g_stop, 0, sizeof g_stop); return 0; }
    for (int i = 0; i < n; i++) {
        int t = (int)strtol(p, &p, 10);
        if (t >= 0 && t < 32768) g_stop[t >> 3] |= (uint8_t)(1u << (t & 7));
    }
    return n;
}

// Fixed text: known tokens, so the weights are read once per batch of up to
// LLM_MAXB tokens (batch 1) or once per token (batch 0, the baseline).
static const float *feed(llm_t *s, const int *ids, int n, int pos, int batch) {
    const float *lg = NULL;
    for (int i = 0; i < n;) {
        int k = batch ? (n - i < LLM_MAXB ? n - i : LLM_MAXB) : 1;
        lg = llm_forward_batch(s, ids + i, pos + i, k);
        if (!lg) return NULL;                                // position or token out of range
        i += k;
    }
    return lg;
}

static int ids_ok(const llm_model_t *m, const int *ids, int n) {
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= m->vocab) return 0;
    return 1;
}

static void run_template(llm_t *s, const llm_model_t *m, char *p) {
    static int seg[MAX_TPL];
    int maxslot = 0, tm = 0, topk = 0, eot = 0, batch = 1, nseg = 0, used = 0;
    unsigned seed = 1;
    if (sscanf(p, "%d %d %d %u %d %d %d%n", &maxslot, &tm, &topk, &seed, &eot, &batch, &nseg, &used) < 7) {
        printf("ERR bad command\n");
        return;
    }
    p += used;
    int nint = 0;
    while (nint < MAX_TPL) {
        char *q = p;
        long v = strtol(p, &p, 10);
        if (p == q) break;
        seg[nint++] = (int)v;
    }
    g_rng = seed ? seed : 1;
    const float *lg = NULL;
    int pos = 0, gen = 0, forced = 0, stop = 0;
    uint64_t cg = 0, cf = 0;
    for (int i = 0, k = 0; k < nseg && i < nint && !stop; k++) {
        if (seg[i] == 0) {                                   // fixed text
            if (i + 1 >= nint) { printf("ERR bad segment\n"); break; }
            int n = seg[i + 1];
            const int *ids = seg + i + 2;
            if (n < 0 || n > nint - i - 2 || !ids_ok(m, ids, n)) { printf("ERR bad segment\n"); break; }
            i += 2 + n;
            if (n == 0) continue;
            if (pos + n > m->ctx) { stop = 1; break; }
            uint32_t a = esp_cpu_get_cycle_count();
            lg = feed(s, ids, n, pos, batch);
            if (!lg) { printf("ERR forward\n"); break; }
            uint32_t dt = esp_cpu_get_cycle_count() - a;
            cf += dt;
            pos += n;
            forced += n;
            printf("Q %d %.2f", n, (double)dt / 240000.0);
            for (int j = 0; j < n; j++) printf(" %d", ids[j]);
            printf("\n");
            fflush(stdout);
        } else if (seg[i] == 1) {                            // slot
            i += 1;
            if (!lg) continue;
            for (int g = 0; g < maxslot && pos < m->ctx; g++) {
                int tok = sample(lg, m->vout, (float)tm / 1000.0f, topk);
                if (tok == eot || is_stop(tok)) break;
                uint32_t a = esp_cpu_get_cycle_count();
                lg = llm_forward(s, tok, pos++);
                uint32_t dt = esp_cpu_get_cycle_count() - a;
                if (!lg) { stop = 1; break; }
                cg += dt;
                gen++;
                printf("T %d %u\n", tok, (unsigned)(dt / 240));
                fflush(stdout);
                if (stop_requested()) { stop = 1; break; }
            }
        } else {
            printf("ERR bad segment\n");
            break;
        }
    }
    printf("R %d %d %.2f %.2f\n", gen, forced, (double)cg / 240000.0, (double)cf / 240000.0);
    fflush(stdout);
}

void demo_loop(llm_t *s, const llm_model_t *m, float head_budget, int psram_mhz) {
    uart_driver_install(UART_NUM_0, 4096, 0, 0, NULL, 0);
    uart_vfs_dev_use_driver(UART_NUM_0);
    llm_set_head_budget(s, head_budget);
    static char line[LINE_MAX_BYTES];
    static int ids[MAX_PROMPT];
    ready(m, head_budget, psram_mhz);
    for (;;) {
        int n = read_line(line, sizeof line);
        if (n == 0) continue;
        if (line[0] == 'I') { ready(m, head_budget, psram_mhz); continue; }
        if (line[0] == 'H') {                        // H budget: share of head rows read (1 = full head)
            float hb = strtof(line + 1, NULL);
            if (hb > 0.0f && hb <= 1.0f) { head_budget = hb; llm_set_head_budget(s, hb); }
            ready(m, head_budget, psram_mhz);
            continue;
        }
        if (line[0] == 'K') { printf("K %d\n", add_stops(line + 1)); fflush(stdout); continue; }
        if (line[0] == 'S') { run_template(s, m, line + 1); continue; }
        if (line[0] != 'G') continue;
        int maxn = 0, tm = 0, topk = 0, eot = 0, np = 0;
        unsigned seed = 1;
        char *p = line + 1;
        int used = 0;
        if (sscanf(p, "%d %d %d %u %d %d%n", &maxn, &tm, &topk, &seed, &eot, &np, &used) < 6 || np < 1) {
            printf("ERR bad command\n");
            continue;
        }
        p += used;
        // A longer prompt is refused, not cut: cutting kept the start and dropped the end
        // (in QA mode, the question).
        int maxp = MAX_PROMPT < m->ctx - 1 ? MAX_PROMPT : m->ctx - 1;
        if (np > maxp) { printf("ERR prompt longer than %d tokens\n", maxp); fflush(stdout); continue; }
        for (int i = 0; i < np; i++) ids[i] = (int)strtol(p, &p, 10);
        if (!ids_ok(m, ids, np)) { printf("ERR token id outside the vocabulary\n"); fflush(stdout); continue; }
        g_rng = seed ? seed : 1;

        uint32_t t0 = esp_cpu_get_cycle_count();
        const float *lg = NULL;
        for (int i = 0; i < np; i += LLM_MAXB) {                // prompt in batches
            int k = np - i < LLM_MAXB ? np - i : LLM_MAXB;
            lg = llm_forward_batch(s, ids + i, i, k);
        }
        if (!lg) { printf("ERR forward\n"); fflush(stdout); continue; }
        printf("P %.1f\n", (double)(uint32_t)(esp_cpu_get_cycle_count() - t0) / 240000.0);
        int pos = np, gen = 0;
        uint64_t cyc = 0;
        uint32_t last_us = 0;
        while (gen < maxn && pos < m->ctx) {
            int tok = sample(lg, m->vout, (float)tm / 1000.0f, topk);
            printf("T %d %u\n", tok, (unsigned)last_us);
            fflush(stdout);
            gen++;
            if (tok == eot || stop_requested()) break;
            uint32_t a = esp_cpu_get_cycle_count();
            lg = llm_forward(s, tok, pos++);
            uint32_t dt = esp_cpu_get_cycle_count() - a;
            if (!lg) break;
            cyc += dt;
            last_us = dt / 240;
        }
        int steps = pos - np;
        printf("E %d %.2f\n", gen, steps ? (double)cyc / 240000.0 / steps : 0.0);
        fflush(stdout);
    }
}
