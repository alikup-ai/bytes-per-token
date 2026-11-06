"""End-to-end engine tests (engine/DESIGN.md, "Tests").

    .venv/Scripts/python engine/test.py qat4_s0                 # PC only
    .venv/Scripts/python engine/test.py qat4_s0 --port COM6     # + board
    options: --flash-app (write the firmware too), --runs N (board repeats)

Checks, each printed PASS/FAIL:
  1  PC engine vs PyTorch (int4 model): NLL gap, argmax agreement, greedy tokens
  2  PC: repeated runs and the two-worker split give identical digests
  3  board: pie2 (PIE, 2 cores) == pie1 (PIE, 1 core) == ref1 (scalar C)
  4  board == PC, bit for bit (token, logit, argmax digests and NLL sum bits)
  5  speed: ms/token per board run, spread across runs, stage breakdown
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
E = ROOT / "engine"
OUT = E / "out"
PY = sys.executable
GCC = r"C:\msys64\ucrt64\bin\gcc.exe"
MODEL_OFFSET = 0x90000    # engine/esp32/partitions.csv
HEAD_BUDGET = 0.15   # must match engine/esp32/main/main.c
results = []


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def eng_lines(text):
    return [json.loads(l[l.index("ENG ") + 4:]) for l in text.splitlines() if "ENG {" in l]


def digests(lines, label):
    d = {}
    for r in lines:
        if r.get("label") != label:
            continue
        if r["test"] == "gen":
            d[f"gen{r['prompt']}"] = (r["tok_digest"], r["logit_digest"])
        elif r["test"] == "eval":
            d["eval"] = (r["nll_sum_bits"], r["argmax_digest"])
        elif r["test"] == "batch":
            d["batch"] = tuple(x["last_logits_digest"] for x in r["runs"])
        elif r["test"] == "rewind":
            d["rewind"] = r["ok"]
    return d


def build_host():
    exe = OUT / "engine_host.exe"
    srcs = [E / "host" / "main.c", E / "src" / "llm.c"]
    if not exe.exists() or max(s.stat().st_mtime for s in srcs + [E / "src" / "llm.h"]) > exe.stat().st_mtime:
        env = dict(os.environ, PATH=r"C:\msys64\ucrt64\bin;" + os.environ["PATH"])
        subprocess.run([GCC, "-O2", "-std=c11", "-Wall", "-Wextra", "-ffp-contract=off", "-static",
                        "-o", str(exe)] + [str(s) for s in srcs], check=True, env=env)
    return exe


def host(exe, pack, *args):
    return subprocess.run([str(exe), str(pack), *args], capture_output=True, text=True, check=True).stdout


def board_run(port, timeout=900):
    import serial
    s = serial.Serial(port, 115200, timeout=1)
    s.dtr, s.rts = False, True
    time.sleep(0.1)
    s.rts = False
    lines, t0, buf = [], time.time(), b""
    while time.time() - t0 < timeout:
        buf += s.readline()                  # may return a partial line on the 1 s timeout
        if not buf.endswith(b"\n"):
            continue                         # long timed tests print one line in pieces
        ln, buf = buf.decode("utf-8", "replace").rstrip(), b""
        if not ln:
            continue
        lines.append(ln)
        if "ENG {" in ln:
            print("    " + ln[:160])
        if '"test":"done"' in ln or '"test":"error"' in ln or "Guru Meditation" in ln:
            break
    s.close()
    return "\n".join(lines)


def keep_awake():
    """Ask Windows not to sleep while this process runs (released when it exits).
    Standby paused the board runs for up to 40 minutes on 2026-09-25."""
    if os.name == "nt":
        import ctypes
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def main():
    keep_awake()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # boot noise from the board can hold bytes cp1251 cannot print
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--port")
    ap.add_argument("--flash-app", action="store_true")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--build", default="build", help="firmware build dir under engine/esp32 (build120 = PSRAM 120 MHz)")
    ap.add_argument("--tag", default="", help="suffix for the board log names")
    a = ap.parse_args()
    pack, refp = OUT / f"{a.name}.pack", OUT / f"{a.name}.ref.json"
    if not pack.exists():
        subprocess.run([PY, str(E / "export.py"), a.name], check=True)
    ref = json.loads(refp.read_text())
    exe = build_host()

    print("1. PC engine vs PyTorch (same int4 weights)")
    amf, lgf = OUT / f"{a.name}.argmax.bin", OUT / f"{a.name}.logits.bin"
    base = host(exe, pack, "--argmax", str(amf), "--logits", str(lgf))
    hl = eng_lines(base)
    ev = next(r for r in hl if r["test"] == "eval")
    gap = ev["nll"] - ref["torch_q4_nll"]
    check("NLL within 0.02 of PyTorch", abs(gap) < 0.02, f"engine {ev['nll']:.4f} torch {ref['torch_q4_nll']:.4f} ({gap:+.4f})")
    agree = float((np.fromfile(amf, dtype=np.int32) == np.array(ref["torch_argmax"])).mean())
    check("argmax agreement >= 97%", agree >= 0.97, f"{agree:.2%} of {ev['targets'] + 2} positions")
    tl = np.load(OUT / f"{a.name}.torch_logits.npy")
    el = np.fromfile(lgf, dtype=np.float32).reshape(tl.shape)
    rows = np.arange(len(tl))
    t1, t2 = tl.argmax(1), np.argsort(tl, 1)[:, -2]
    gap_err = np.abs((el[rows, t1] - el[rows, t2]) - (tl[rows, t1] - tl[rows, t2]))
    tie = float(np.percentile(gap_err, 99))
    check("logits close to PyTorch (int8 activations)", tie < 0.3,
          f"|dlogit| median {np.median(np.abs(el - tl)):.3f}; top-2 gap error median {np.median(gap_err):.3f}, "
          f"p99 {tie:.3f} (logit std {tl.std():.2f})")
    gens = [r["ids"] for r in hl if r["test"] == "gen"]
    same = [next((i for i, (x, y) in enumerate(zip(g, t)) if x != y), len(g)) for g, t in zip(gens, ref["torch_gen"])]
    # A divergence is only a defect if PyTorch was decisive there. Where torch's
    # own top-2 gap is below the engine's measured gap error (p99, above), int8
    # activations can legitimately flip the choice.
    gaps = [ref["torch_gen_margin"][k][i] for k, i in enumerate(same) if i < len(gens[k])]
    check("greedy generations match PyTorch up to near ties", all(g < max(tie, 0.05) for g in gaps),
          f"identical prefix per prompt: {same} of {len(gens[0])}; torch top-2 gap where they split: "
          + (", ".join(f"{g:.4f}" for g in gaps) or "none"))

    print("2. PC determinism, with the board's head budget")
    hb = ["--head-budget", str(HEAD_BUDGET)]
    hl2 = eng_lines(host(exe, pack, *hb))
    hd = digests(hl2, "host")
    ev2 = next(r for r in hl2 if r["test"] == "eval")
    if "screen_top1_agree" in ev2:
        check("two-stage head: read rows bit-identical to full head", ev2["screen_read_mismatch"] == 0,
              f"{ev2['screen_rows']:.0f} of {ref['cfg']['vocab']} rows read, top-1 same as full head {ev2['screen_top1_agree']:.2%}")
        check("two-stage head: top-1 agreement >= 98%", ev2["screen_top1_agree"] >= 0.98)
    again = digests(eng_lines(host(exe, pack, *hb)), "host")
    split = {k: v for k, v in digests(eng_lines(host(exe, pack, "--split", *hb)), "host-split").items()}
    bl_ = next(r for r in hl2 if r["test"] == "batch")
    dg = {x["last_logits_digest"] for x in bl_["runs"]}
    check("batched = sequential (batch 1..16 give identical logits)", len(dg) == 1,
          ", ".join(f"b{x['batch']} {x['last_logits_digest']}" for x in bl_["runs"]))
    rw = next((r for r in hl2 if r["test"] == "rewind"), None)
    check("rewind: stepping back reproduces the logits", rw is not None and rw["ok"] == 1,
          f"positions {rw['from']} -> {rw['to']}, {rw['redone']} redone" if rw else "no rewind line")
    check("two PC runs identical", again == hd)
    check("two-worker split identical", split == hd)

    if a.port:
        if a.flash_app:
            subprocess.run([PY, "-m", "esptool", "--chip", "esp32s3", "-p", a.port, "-b", "921600", "write_flash",
                            "@flash_args"], cwd=E / "esp32" / a.build, check=True)
        import hashlib
        stamp = OUT / f".flashed_{a.port}"
        want = f"{hashlib.sha256(pack.read_bytes()).hexdigest()} {MODEL_OFFSET:#x}"
        if a.flash_app or not stamp.exists() or stamp.read_text() != want:   # a 15 MB pack takes ~3 min
            subprocess.run([PY, "-m", "esptool", "--chip", "esp32s3", "-p", a.port, "-b", "921600", "write_flash",
                            hex(MODEL_OFFSET), str(pack)], check=True, capture_output=True)
            stamp.write_text(want)
        speeds = {"pie2": [], "pie1": [], "pie2full": [], "pie2warm": []}
        stages = None
        for run in range(a.runs):
            print(f"3-5. board run {run + 1}/{a.runs}")
            log = board_run(a.port)
            (OUT / f"{a.name}{a.tag}.board{run + 1}.log").write_text(log, encoding="utf-8")
            bl = eng_lines(log)
            cfg = next((r for r in bl if r["test"] == "config"), None)
            if cfg is None or not any(r["test"] == "done" for r in bl):
                check("board run completed", False, log[-400:])
                break
            if run == 0:
                print(f"    weights: SRAM {cfg['weights_sram'] / 1024:.0f} KB, PSRAM {cfg['weights_psram'] / 1024:.0f} KB, "
                      f"flash in place {cfg['flash_inplace'] / 1024:.0f} KB; load {cfg['load_ms']} ms")
            d2, d1, d0 = digests(bl, "pie2"), digests(bl, "pie1"), digests(bl, "ref1")
            check(f"run {run + 1}: pie2 == pie1 == ref1", d2 == d1 == d0 and len(d2) == len(hd))
            check(f"run {run + 1}: board == PC (bit-exact)", d2 == hd,
                  "" if d2 == hd else json.dumps({k: (d2.get(k), hd[k]) for k in hd if d2.get(k) != hd[k]}))
            for r in bl:
                if r["test"] == "speed":
                    speeds[r["label"]].append(r["ms_per_tok"])
                    if r["label"] == "pie2":
                        stages = r["stage_ms_per_tok"]
                if r["test"] == "poscurve" and run == 0:
                    print(f"  {r['label']} ms/token by position (32 per bucket, from {r['first_pos']}): "
                          + ", ".join(f"{v:.2f}" for v in r["ms_per_tok"]))
        for lab, v in speeds.items():
            if v:
                v = np.array(v)
                print(f"  speed {lab}: {v.mean():.2f} ms/token = {1000 / v.mean():.1f} tok/s "
                      f"(min {v.min():.2f}, max {v.max():.2f}, spread {100 * (v.max() - v.min()) / v.mean():.1f}%, n={len(v)})")
            bt = next((r for r in bl if r["test"] == "batch" and r["label"] == "pie2"), None)
            if bt and run == 0:
                print("  batch (64 known tokens) ms/token: " + ", ".join(f"n={x['batch']} {x['ms_per_tok']:.2f}" for x in bt["runs"]))
        if stages:
            print("  pie2 stages ms/token: " + ", ".join(f"{k} {v:.2f}" for k, v in stages.items()))
    bad = [n for n, ok in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} checks passed" + (f"; FAILED: {bad}" if bad else ""))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
