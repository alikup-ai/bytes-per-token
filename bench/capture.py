"""Capture the bench firmware's BENCH lines and write roofline.json.

    .venv/Scripts/python bench/capture.py --port COM5          # read the board
    .venv/Scripts/python bench/capture.py --log bench/run1.log # or a saved log

Resets the board through DTR/RTS (same as idf.py monitor), reads until
BENCH {"test":"done"}, saves the raw log next to roofline.json.
"""
import argparse
import json
import time
from pathlib import Path

HERE = Path(__file__).parent


def read_port(port, timeout_s=600):
    import serial
    s = serial.Serial(port, 115200, timeout=1)
    s.dtr, s.rts = False, True   # hold in reset
    time.sleep(0.1)
    s.rts = False                # release: boots the app
    lines, t0 = [], time.time()
    while time.time() - t0 < timeout_s:
        line = s.readline().decode("utf-8", "replace").rstrip()
        if not line:
            continue
        print(line)
        lines.append(line)
        if '"test":"done"' in line:
            break
    return lines


def parse(lines):
    out = {"seq_read": [], "flash_row": [], "gemv": [], "gemv_dual": [], "dma_overlap": [], "boot": []}
    for ln in lines:
        i = ln.find("BENCH ")
        if i < 0:
            continue
        rec = json.loads(ln[i + 6:])
        out.setdefault(rec.pop("test"), []).append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--log")
    ap.add_argument("--out", default=str(HERE / "roofline.json"))
    a = ap.parse_args()
    if a.port:
        lines = read_port(a.port)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        (HERE / f"run-{stamp}.log").write_text("\n".join(lines), encoding="utf-8")
    else:
        lines = Path(a.log).read_text(encoding="utf-8", errors="replace").splitlines()
    res = parse(lines)
    if not res.get("done"):
        print("WARNING: no 'done' line; the run was cut short")
    bad = [g for g in res["gemv"] if not g.get("match_ref")]
    if bad:
        print(f"WARNING: {len(bad)} gemv results do not match the scalar reference")
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
