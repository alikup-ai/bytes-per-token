"""Summarize H2 runs: quality vs weight bytes read per token vs predicted
latency on the ESP32-S3.

    .venv/Scripts/python train/report_h2.py

Latency is predicted twice with the phase-0 cost model: on the measured
esp32-ai C engine (fitted, validated to ~5%), and on the 'optimized' what-if
engine (PIE 1 cyc/MAC, int4, int8 KV, dual core -- assumptions until the
phase-1 bench replaces them). Mean position 128.
"""
import copy
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "costmodel"))
from arch import Arch  # noqa: E402
from validate import fit, load  # noqa: E402
from whatif import cost  # noqa: E402
import whatif  # noqa: E402

POS = 128


def arch_of(cfg):
    return Arch(vin=cfg["vocab"], vout=cfg["vocab"], d=cfg["d"], layers=cfg["layers"],
                heads=cfg["heads"], ffn=cfg["ffn"], ple=cfg["ple"], experts=cfg["experts"])


def main():
    hw, engines, runs = load()
    theta = fit("esp32ai-c-int8", runs, hw, engines)
    base = engines["esp32ai-c-int8"]
    opt = copy.deepcopy(base)
    opt.update(dual_all=True, attn_cores=2, kv_bytes=1)
    opt["core"]["bits"] = opt["head"]["bits"] = 4
    th_opt = theta.copy()
    th_opt[0], th_opt[1], th_opt[3] = 1.0, 1.0, 10.0

    whatif.POS = POS
    rows = []
    for f in sorted((ROOT / "runs").glob("*/final.json")):
        m = json.loads(f.read_text())
        whatif.ARCHS = {"x": arch_of(m["cfg"])}
        whatif.ARCH = "x"
        ms_now = sum(v[2] for v in cost(theta, base, hw).values())
        ms_opt = sum(v[2] for v in cost(th_opt, opt, hw).values())
        rows.append((m["name"], m["stored_params"] / 1e6, m["bytes_per_token_4bit"] / 1e3,
                     m["final_val_loss"], ms_now, ms_opt, m["tokens"] / 1e6))
    print(f"{'run':16s} {'stored M':>8s} {'KB/tok@4b':>9s} {'val loss':>8s} {'ppl':>6s} "
          f"{'ms/tok now':>10s} {'ms/tok opt':>10s} {'train Mtok':>10s}")
    for r in rows:
        print(f"{r[0]:16s} {r[1]:8.2f} {r[2]:9.0f} {r[3]:8.4f} {math.exp(r[3]):6.2f} "
              f"{r[4]:10.1f} {r[5]:10.1f} {r[6]:10.1f}")


if __name__ == "__main__":
    main()
