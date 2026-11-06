// PC runner: the same engine and test sequence as the board, on the scalar
// reference kernels. Output lines must match the board's digests exactly.
//
//   engine_host <name>.pack [--split] [--argmax file] [--gen N] [--eval N] [--repeats N]
//
// --split runs every parallel section as two slices (worker 0, then worker 1),
// the way the board's two cores divide it, to check that splitting changes nothing.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/llm.h"

static void *alloc(size_t n, int fast) {
    (void)fast;
    uint8_t *p = malloc(n + 64);
    if (!p) return NULL;
    memset(p, 0, n + 64);
    return (void *)(((uintptr_t)p + 63) & ~(uintptr_t)63);   // leaked on purpose: lives until exit
}

static const void *place(const char *name, const void *src, size_t n, int kind) {
    (void)name; (void)n; (void)kind;
    return src;
}

static uint32_t usec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint32_t)((uint64_t)t.tv_sec * 1000000u + (uint64_t)t.tv_nsec / 1000u);
}

static void par_split(par_body_fn fn, void *ctx, int n) {
    fn(ctx, 0, n / 2, 0);
    fn(ctx, n / 2, n, 1);
}

static int out(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vprintf(fmt, ap);
    va_end(ap);
    return r;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.pack [--split] [--argmax f] [--gen N] [--eval N]\n", argv[0]); return 2; }
    int split = 0, gen = -1, eval = -1, repeats = 0;
    float hb = 0.0f;
    const char *amf = NULL, *lgf = NULL;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--split")) split = 1;
        else if (!strcmp(argv[i], "--argmax") && i + 1 < argc) amf = argv[++i];
        else if (!strcmp(argv[i], "--logits") && i + 1 < argc) lgf = argv[++i];
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc) gen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eval") && i + 1 < argc) eval = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--repeats") && i + 1 < argc) repeats = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--head-budget") && i + 1 < argc) hb = (float)atof(argv[++i]);
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *blob = alloc((size_t)size, 0);
    if (fread(blob, 1, (size_t)size, f) != (size_t)size) { perror("read"); return 1; }
    fclose(f);

    llm_platform_t p = {llm_ref_q4g, llm_ref_s8, llm_ref_colsum, split ? par_split : NULL, usec, place};
    llm_model_t m;
    int e = llm_load(&m, blob, (size_t)size, &p);
    if (e) { fprintf(stderr, "load failed: %d\n", e); return 1; }
    llm_t *s = llm_new(&m, &p, alloc);
    if (!s) { fprintf(stderr, "alloc failed\n"); return 1; }
    int32_t *am = alloc(sizeof(int32_t) * (size_t)m.n_eval, 0);
    llm_run_t r = {split ? "host-split" : "host", m.n_prompts, gen >= 0 ? gen : m.n_gen,
                   eval >= 0 ? eval : m.n_eval, repeats, hb, 64, am, NULL, 1, 0};
    if (lgf) r.logits_out = alloc(sizeof(float) * (size_t)m.ctx * m.vout, 0);
    out("ENG {\"label\":\"%s\",\"test\":\"config\",\"vocab\":%d,\"d\":%d,\"layers\":%d,\"heads\":%d,"
        "\"ffn\":%d,\"ple\":%d,\"act\":%d,\"ctx\":%d}\n", r.label, m.vocab, m.d, m.layers, m.heads,
        m.ffn, m.ple, m.act, m.ctx);
    llm_run_tests(s, &m, &r, out, 1.0);
    if (amf) {
        FILE *g = fopen(amf, "wb");
        fwrite(am, sizeof(int32_t), (size_t)r.eval, g);
        fclose(g);
    }
    if (lgf) {                       // first eval window, ctx x vocab floats
        FILE *g = fopen(lgf, "wb");
        fwrite(r.logits_out, sizeof(float), (size_t)m.ctx * m.vout, g);
        fclose(g);
    }
    return 0;
}
