"""Run slvDev/esp32-ai's own TinyStories firmware on our board and record its
self-reported speed (docs/phase3-esp32ai.md, re-measurement).

    .venv/Scripts/python bench/esp32ai_capture.py COM6 [runs]

Resets the board through DTR/RTS, reads until the firmware's
"profile ms/token" line, and checks the boot fingerprint against the model file.
Their ms/token is decode time inside llm_forward only (printing excluded).
"""
import json
import re
import sys
import time
from pathlib import Path

import serial

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "refs" / "esp32-ai" / "artifacts" / "tinystories" / "model.bin"


def fnv(data):
    h = 2166136261
    for b in data:
        h = ((h ^ b) * 16777619) & 0xFFFFFFFF
    return f"{h:08x}"


def run(port):
    s = serial.Serial(port, 115200, timeout=1)
    s.dtr, s.rts = False, True
    time.sleep(0.1)
    s.rts = False
    lines, buf, t0 = [], b"", time.time()
    while time.time() - t0 < 300:
        buf += s.readline()
        if not buf.endswith(b"\n"):
            continue
        ln, buf = buf.decode("utf-8", "replace").rstrip(), b""
        lines.append(ln)
        if ln.startswith("profile ms/token") or "FATAL" in ln or "Guru" in ln:
            break
    s.close()
    return lines


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "COM6"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    want = fnv(MODEL.read_bytes())
    res = []
    for i in range(n):
        lines = run(port)
        (ROOT / "bench" / f"esp32ai_fw_run{i + 1}.log").write_text("\n".join(lines), encoding="utf-8")
        txt = "\n".join(lines)
        fp = re.search(r"build: bytes=(\d+) fp=([0-9a-f]{8})", txt)
        tp = re.search(r"--- (\d+) tokens in ([\d.]+) s ---\s*\nthroughput: ([\d.]+) tok/s\s+\(([\d.]+) ms/token\)", txt)
        pr = re.search(r"profile ms/token: (.*)", txt)
        r = dict(run=i + 1, fingerprint=fp.group(2) if fp else None, fingerprint_ok=bool(fp and fp.group(2) == want),
                 tokens=int(tp.group(1)) if tp else None, wall_tok_s=float(tp.group(3)) if tp else None,
                 ms_per_token=float(tp.group(4)) if tp else None, profile=pr.group(1) if pr else None)
        res.append(r)
        print(json.dumps(r, ensure_ascii=False))
    ms = [r["ms_per_token"] for r in res if r["ms_per_token"]]
    if ms:
        print(f"esp32-ai firmware on this board: {sum(ms) / len(ms):.2f} ms/token = {1000 * len(ms) / sum(ms):.2f} tok/s "
              f"(runs {len(ms)}, spread {max(ms) - min(ms):.2f} ms)")


if __name__ == "__main__":
    main()
