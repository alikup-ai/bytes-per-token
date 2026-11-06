"""What would each lever buy? Calibrated on esp32-ai's measured TinyStories
profile, then re-costed under hypothetical changes.

Every scenario below that changes cycles/MAC is an ASSUMPTION until bench/
measures PIE throughput on the board (plan phase 1). The point of this table
is ordering and rough size, not a promise.

    python costmodel/whatif.py
"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arch import ARCHS, ops  # noqa: E402
from model import STAGES, op_terms  # noqa: E402
from validate import fit, load  # noqa: E402

ENGINE, ARCH, POS, BUS = "esp32ai-c-int8", "tinystories", 103.5, "psram@80"


def cost(theta, eng, hw, *, overlap=False):
    """Per-stage (compute_ms, memory_ms, total_ms). overlap=True models DMA
    double-buffering: each op costs max(compute, memory) instead of the sum."""
    out = {s: [0.0, 0.0, 0.0] for s in STAGES}
    for op in ops(ARCHS[ARCH], eng, POS):
        c, m, k = op_terms(op, hw, BUS)
        comp = c @ theta
        mem = m @ theta + k
        tot = max(comp, mem) if overlap else comp + mem
        row = out[op.stage]
        row[0] += comp * 1e3
        row[1] += mem * 1e3
        row[2] += tot * 1e3
    return out


def main():
    hw, engines, runs = load()
    theta0 = fit(ENGINE, runs, hw, engines)
    base_eng = engines[ENGINE]

    def scenario(**kw):
        th, eng = theta0.copy(), copy.deepcopy(base_eng)
        if "cyc_mac" in kw:
            th[0] = kw["cyc_mac"]
        if "cyc_attn" in kw:
            th[1] = kw["cyc_attn"]
        if "call_us" in kw:
            th[3] = kw["call_us"]
        if "wbits" in kw:
            eng["core"]["bits"] = eng["head"]["bits"] = kw["wbits"]
        if "kv_bytes" in kw:
            eng["kv_bytes"] = kw["kv_bytes"]
        if kw.get("attn_dual"):
            eng["attn_cores"] = 2
        if kw.get("all_dual"):
            eng["dual_all"] = True
        return cost(th, eng, hw, overlap=kw.get("overlap", False))

    rows = [
        ("0 baseline (esp32-ai, measured 94.9 ms)", {}),
        ("1 PIE matvec, 1.0 cyc/MAC [assumed]", dict(cyc_mac=1.0)),
        ("2 + all matvecs & attention on 2 cores", dict(cyc_mac=1.0, all_dual=True, attn_dual=True)),
        ("3 + weights int4 in PSRAM (unpack in PIE)", dict(cyc_mac=1.0, all_dual=True, attn_dual=True, wbits=4)),
        ("4 + KV cache int8, PIE attention 1.0 cyc", dict(cyc_mac=1.0, all_dual=True, attn_dual=True, wbits=4,
                                                         kv_bytes=1, cyc_attn=1.0)),
        ("5 + call overhead 80 -> 10 us", dict(cyc_mac=1.0, all_dual=True, attn_dual=True, wbits=4,
                                               kv_bytes=1, cyc_attn=1.0, call_us=10)),
        ("6 + DMA overlap: max(compute, memory)", dict(cyc_mac=1.0, all_dual=True, attn_dual=True, wbits=4,
                                                       kv_bytes=1, cyc_attn=1.0, call_us=10, overlap=True)),
        ("7 = 6 with 2-bit weights", dict(cyc_mac=1.0, all_dual=True, attn_dual=True, wbits=2,
                                          kv_bytes=1, cyc_attn=1.0, call_us=10, overlap=True)),
    ]
    print(f"{ARCH} on {ENGINE}, mean position {POS}, {BUS}")
    print(f"{'scenario':46s} {'ms/tok':>7s} {'tok/s':>6s} {'compute':>8s} {'memory':>7s}  biggest stage")
    for name, kw in rows:
        st = scenario(**kw)
        tot = sum(v[2] for v in st.values())
        comp = sum(v[0] for v in st.values())
        mem = sum(v[1] for v in st.values())
        big = max(st, key=lambda s: st[s][2])
        print(f"{name:46s} {tot:7.1f} {1000 / tot:6.1f} {comp:8.1f} {mem:7.1f}  {big} {st[big][2]:.1f}")


if __name__ == "__main__":
    main()
