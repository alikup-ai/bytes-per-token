#include <math.h>
#include <stdio.h>
#include "../src/llm.h"
int main(void) {
    double me = 0, mr = 0, ms = 0, ml = 0;
    for (int i = 0; i <= 2000000; i++) {
        float x = -80.0f + 160.0f * i / 2000000.0f;
        double e = fabs(llm_exp(x) / exp((double)x) - 1); if (e > me) me = e;
        float y = 1e-6f * powf(1e12f, (float)i / 2000000.0f);
        double r = fabs(llm_recip(y) * (double)y - 1); if (r > mr) mr = r;
        double q = fabs(llm_rsqrt(y) * sqrt((double)y) - 1); if (q > ms) ms = q;
        double l = fabs(llm_log(y) - log((double)y)); if (l > ml) ml = l;
    }
    printf("max rel err: exp %.2e  recip %.2e  rsqrt %.2e | max abs err log %.2e\n", me, mr, ms, ml);
    return 0;
}
