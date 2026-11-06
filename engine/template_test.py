"""Template filling on the board (plan item 3: skip tokens fixed by a grammar).

The fixed text of a template is known, so the engine feeds it in batches and
the model only generates the slots. Each run does the same template twice:
batch 1 (fixed text in batches of 16) and batch 0 (token by token, the
baseline). Greedy decoding, so both must produce the same tokens; only the time
may differ.

    .venv/Scripts/python engine/template_test.py --port COM6 [--runs 3]

Needs the demo firmware (engine/demo/flash_demo.sh). Slots are written as ___;
a slot ends when the model wants a token that starts with punctuation (the
template supplies it), at end of text, or after --maxslot tokens.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import serial
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
TOK = {4096: ROOT / "data" / "tok4096.json", 8192: ROOT / "data" / "tok8192.json",
       16384: ROOT / "data" / "tok16384.json", 32768: ROOT / "engine" / "out" / "esp32ai" / "tokenizer.json"}
TEMPLATES = [
    "Once upon a time, there was a little ___. She lived in ___. Every day, she ___. "
    "One day, ___. She said, \"___!\" In the end, ___.",
    "Tom had a big red ___. He liked to ___. His mom said, \"___.\" Tom was ___. "
    "The next day, ___. From that day on, ___.",
]
PUNCT = ".,!?;:\""


def stop_ids(tok, vocab):
    out = []
    for i in range(vocab):
        t = tok.decode([i])
        u = t.lstrip(" ")
        if "\n" in t or (u and u[0] in PUNCT):
            out.append(i)
    return out


def segments(tok, text):
    segs, parts = [], text.split("___")
    for k, part in enumerate(parts):
        if k < len(parts) - 1:
            part = part.rstrip(" ")        # the space belongs to the slot's first word in byte-level BPE
        if part:
            segs.append([0, *tok.encode(part).ids])
        if k < len(parts) - 1:
            segs.append([1])
    return segs


class Board:
    def __init__(self, port):
        self.s = serial.Serial(port, 115200, timeout=1)
        self.s.dtr = self.s.rts = False

    def send(self, line):
        self.s.write((line + "\n").encode())

    def line(self, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            ln = self.s.readline().decode("utf-8", "replace").strip()
            if ln:
                return ln
        raise TimeoutError("board silent")

    def ready(self):
        for _ in range(5):
            self.send("I")
            t0 = time.time()
            while time.time() - t0 < 3:
                ln = self.s.readline().decode("utf-8", "replace").strip()
                if ln.startswith("READY "):
                    return json.loads(ln[6:])
        raise TimeoutError("no READY (is the demo firmware flashed?)")


def run(b, segs, batch, maxslot, eot):
    flat = [x for sg in segs for x in (sg if sg[0] == 1 else [0, len(sg) - 1, *sg[1:]])]
    b.send(f"S {maxslot} 0 1 1 {eot} {batch} {len(segs)} " + " ".join(map(str, flat)))
    toks, gen = [], []
    while True:
        ln = b.line()
        if ln.startswith("Q "):
            toks += [int(x) for x in ln.split()[3:]]
        elif ln.startswith("T "):
            t = int(ln.split()[1])
            toks.append(t)
            gen.append(t)
        elif ln.startswith("R "):
            g, f, mg, mf = ln.split()[1:]
            return toks, gen, int(g), int(f), float(mg), float(mf)
        elif ln.startswith("ERR"):
            raise RuntimeError(ln)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--maxslot", type=int, default=24)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    b = Board(a.port)
    info = b.ready()
    tok = Tokenizer.from_file(str(TOK[info["vocab"]]))
    stops = stop_ids(tok, info["vout"])
    b.send("K 0"); b.line()
    for i in range(0, len(stops), 300):
        chunk = stops[i:i + 300]
        b.send(f"K {len(chunk)} " + " ".join(map(str, chunk))); b.line()
    print(f"model d={info['d']} L={info['layers']} vocab={info['vocab']} PSRAM {info['psram_mhz']} MHz, "
          f"{len(stops)} stop tokens")

    ok = True
    for text in TEMPLATES:
        segs = segments(tok, text)
        res = {}
        for batch in (1, 0):
            best = None
            for _ in range(a.runs):
                r = run(b, segs, batch, a.maxslot, 0)
                if best and r[1] != best[1]:
                    print(f"  NONDETERMINISTIC (batch {batch})"); ok = False
                if not best or r[4] + r[5] < best[4] + best[5]:
                    best = r
            res[batch] = best
        same = res[1][0] == res[0][0]
        ok &= same
        toks, gen, g, f, mg1, mf1 = res[1]
        mg0, mf0 = res[0][4], res[0][5]
        t1, t0 = mg1 + mf1, mg0 + mf0
        print("\n" + tok.decode(toks))
        print(f"  tokens: {g} generated + {f} from the template ({f / (g + f):.0%} fixed)")
        print(f"  token by token: {t0:7.1f} ms  (slots {mg0:.1f} + fixed {mf0:.1f})")
        print(f"  batched fixed:  {t1:7.1f} ms  (slots {mg1:.1f} + fixed {mf1:.1f})  -> x{t0 / t1:.2f} faster")
        print(f"  free generation of the same {g + f} tokens would take ~{(g + f) * mg1 / max(g, 1):.1f} ms "
              f"-> template x{(g + f) * mg1 / max(g, 1) / t1:.2f}")
        print(f"  same tokens in both modes: {'yes' if same else 'NO'}")
    print("\nALL OK" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
