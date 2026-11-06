"""New ways to compute, costed on paper before any of them is built.

Starts from whatif.py scenario 5 (PIE 1 cyc/MAC, int4, int8 KV, both cores,
10 us/call) and applies one idea at a time, with and without DMA overlap
(overlap is unverified until the board bench runs). Every knob marked
[assumed] is a guess from the literature, not a measurement on the S3.
Quality is NOT modelled here: each idea needs its own training run.

    python costmodel/ideas.py [--pos N]     # N: mean decode position (default 128)

The last table checks the MEAS column against the board at the positions the
board actually ran (docs/phase3-engine.md, phase3-esp32ai.md). Until 2026-09-29
the docs compared the position-128 prediction with a measurement at positions
~13-61, which read as +4%. At matched positions (and with the fp32 scale per
64 weights our engine stores) the model is 10-14% low.
"""
import argparse
import copy
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arch import ARCHS, Arch, Op, ops  # noqa: E402
from model import op_terms  # noqa: E402
from validate import fit, load  # noqa: E402

POS, BUS = 128, "psram@80"
# H2 dense arm from train/sweep_h2.sh: the size we actually train today.
ARCHS_HERE = {
    "tinystories 28.9M (esp32-ai)": ARCHS["tinystories"],
    "h2 dense 1.8M (ours)": Arch(vin=4096, vout=4096, d=128, layers=6, heads=4, ffn=384, ple=0),
}
# Scale group of the int4 weights on our engine: our models 64 (fp32 scale per 64
# weights); esp32-ai's released weights keep one scale per row (0).
GROUP = {"tinystories 28.9M (esp32-ai)": 0, "h2 dense 1.8M (ours)": 64}

# Board measurements (PIE, 2 cores, PSRAM 80 MHz, full head, no idea applied) at their
# own mean decode position. NB: the board keeps ~186-210 KB of weights in SRAM, which
# the model prices as PSRAM, so the true model error is a little larger than shown.
BOARD = [
    ("h2 dense 1.8M (ours)", 36.5, 16.55, "qat4, engine v1, prompt 0 + 48 tokens (pos 13..60)"),
    ("tinystories 28.9M (esp32-ai)", 36.5, 24.48, "engine v2, prompt 0 + 48 tokens"),
    ("tinystories 28.9M (esp32-ai)", 103.5, 25.00, "engine v3, their bench, pos 4..203"),
]


def is_weight(op):
    return op.kind == "matvec"


def transform(a, eng, k):
    """Per-token op list with idea knobs applied. Returns (ops, extra_cycles)."""
    out, extra = [], 0.0
    for op in ops(a, eng, POS):
        op = copy.copy(op)
        if op.stage == "ffn" and is_weight(op) and "ffn_frac" in k:
            op.macs *= k["ffn_frac"]
            op.bytes *= k["ffn_frac"]
        if op.stage == "head" and "head_frac" in k:
            op.macs *= k["head_frac"]
            op.bytes *= k["head_frac"]
        if is_weight(op) and "byte_ratio" in k:
            op.bytes *= k["byte_ratio"]
            extra += op.macs * k["cyc_per_weight"] / op.cores
        out.append(op)
    bits = eng["core"]["bits"]
    if "ffn_frac" in k and not k.get("gate_first"):
        # low-rank activity predictor per layer: D -> r -> F
        r = a.d // 4
        for _ in range(a.layers):
            n = a.d * r + r * a.ffn
            out.append(Op("ffn", "matvec", macs=n, bytes=n * bits / 8, tier="psram", cores=2, calls=2))
    if "head_frac" in k:
        # two-stage head: score sqrt(V) cluster centroids, then only rows of the top clusters
        c = int(math.sqrt(a.vout))
        out.append(Op("head", "matvec", macs=c * a.d, bytes=c * a.d * bits / 8, tier="psram", cores=2, calls=1))
    return out, extra


def ms_per_token(a, eng, theta, hw, k, overlap):
    olist, extra = transform(a, eng, k)
    comp_s = mem_s = tot_s = 0.0
    for op in olist:
        c, m, kk = op_terms(op, hw, BUS)
        comp, mem = c @ theta, m @ theta + kk
        comp_s += comp
        mem_s += mem
        tot_s += max(comp, mem) if overlap else comp + mem
    x = extra / hw["f_hz"]
    comp_s += x
    tot_s += x  # decode/generation work sits on the compute side; charged in full
    n = k.get("verify", 1)       # speculative: verify n tokens in one weight pass
    if n > 1:
        # weights read once, compute done n times; accept `accept` tokens per pass
        comp_n = comp_s * n
        tot_s = max(comp_n, mem_s) if overlap else comp_n + mem_s
        tot_s /= k["accept"]
    tot_s *= 1 - k.get("free_frac", 0.0)
    return tot_s * 1e3, comp_s * 1e3, mem_s * 1e3


IDEAS = [
    ("base: scenario 5 (PIE, int4, int8 KV, 2 cores)", {}),
    ("1 skip zero FFN neurons, 20% read [assumed]", dict(ffn_frac=0.2)),
    ("1' ReLU gate-first, exact: 48% of FFN read [measured]", dict(ffn_frac=0.482, gate_first=True)),
    ("1b two-stage head, 10% of rows read [assumed]", dict(head_frac=0.1)),
    ("1+1b both", dict(ffn_frac=0.2, head_frac=0.1)),
    ("2 weights from seeds (SeedLM), 2x fewer bytes, 4 cyc/w", dict(byte_ratio=0.5, cyc_per_weight=4)),
    ("2' same, 1 cyc/w (PIE generator)", dict(byte_ratio=0.5, cyc_per_weight=1)),
    ("3 2-bit + lookup tables, 0.5 cyc/MAC [assumed]", dict(bits=2, cyc_mac=0.5)),
    ("4 entropy-coded int4 (~3.3 bits), 3 cyc/w decode", dict(byte_ratio=0.82, cyc_per_weight=3)),
    ("5a speculative, verify 4, accept 2.0 [assumed]", dict(verify=4, accept=2.0)),
    ("5b JSON: 30% of tokens forced by schema [assumed]", dict(free_frac=0.3)),
    ("combo 1+1b + 2-bit LUT + 5a", dict(ffn_frac=0.2, head_frac=0.1, bits=2, cyc_mac=0.5,
                                          verify=4, accept=2.0)),
]


# Board bench, 2026-09-25 (bench/roofline_run*.json, 5 runs, spread < 0.1%):
# PIE int4 gemv from SRAM 0.469 cyc/MAC; streamed from PSRAM the same kernel
# reaches 73.1 MB/s of weights on one core, 83.4 on two (bus: 86.7 / 89.8), so
# the kernel already hides its compute under the memory stream: max(), not sum.
# 256 B flash row: 9.66 us. Attention PIE and per-call cost are still assumed.
MEASURED = dict(cyc_mac=0.469, psram_Bps={"1": 73.1e6, "2": 83.4e6}, flash_row_us=9.66)


def main():
    global POS
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", type=float, default=POS)
    POS = ap.parse_args().pos
    hw, engines, runs = load()
    hw_m = copy.deepcopy(hw)
    hw_m["bw_Bps"][BUS] = MEASURED["psram_Bps"]
    hw_m["flash_row_us"] = MEASURED["flash_row_us"]
    theta0 = fit("esp32ai-c-int8", runs, hw, engines)
    base = copy.deepcopy(engines["esp32ai-c-int8"])
    base.update(dual_all=True, attn_cores=2, kv_bytes=1)
    base["core"]["bits"] = base["head"]["bits"] = 4
    for aname, a in ARCHS_HERE.items():
        base["core"]["group"] = base["head"]["group"] = GROUP[aname]
        print(f"\n== {aname}, pos {POS}, {BUS}")
        print(f"{'idea':56s} {'ms seq':>7s} {'x':>5s} {'ms ovl':>7s} {'x':>5s}  comp/mem ms"
              f" | {'ms MEAS':>7s} {'x':>5s}  comp/mem")
        ref = None
        for name, k in IDEAS:
            th, eng = theta0.copy(), copy.deepcopy(base)
            th[0], th[1], th[3] = k.get("cyc_mac", 1.0), 1.0, 10.0
            if "bits" in k:
                eng["core"]["bits"] = eng["head"]["bits"] = k["bits"]
            seq, comp, mem = ms_per_token(a, eng, th, hw, k, False)
            ovl, _, _ = ms_per_token(a, eng, th, hw, k, True)
            th_m = th.copy()
            th_m[0] = min(th[0], MEASURED["cyc_mac"])
            meas, comp_m, mem_m = ms_per_token(a, eng, th_m, hw_m, k, True)
            if ref is None:
                ref = (seq, ovl, meas)
            print(f"{name:56s} {seq:7.1f} {ref[0] / seq:5.2f} {ovl:7.1f} {ref[1] / ovl:5.2f}  "
                  f"{comp:.1f}/{mem:.1f} | {meas:7.1f} {ref[2] / meas:5.2f}  {comp_m:.1f}/{mem_m:.1f}")

    print("\n== MEAS column against the board, at the board's own positions")
    th_m = theta0.copy()
    th_m[0], th_m[1], th_m[3] = MEASURED["cyc_mac"], 1.0, 10.0
    for aname, pos, meas, note in BOARD:
        base["core"]["group"] = base["head"]["group"] = GROUP[aname]
        POS = pos
        pred, _, _ = ms_per_token(ARCHS_HERE[aname], base, th_m, hw_m, {}, True)
        print(f"   {aname:30s} pos {pos:6.1f}: model {pred:5.2f} ms, board {meas:5.2f} ms "
              f"(model {pred / meas - 1:+.1%})  {note}")


if __name__ == "__main__":
    main()
