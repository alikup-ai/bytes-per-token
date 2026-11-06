"""How many tokens would one weight pass yield? Offline simulation of drafters
against the model's own greedy output (plan phase 4, H3).

    .venv/Scripts/python decode/spec_sim.py combo_s0

Greedy continuations come from the int4 PyTorch model (what the engine runs,
up to int8 noise). A pass verifies up to D drafted tokens; it yields
1 + (number of leading drafts equal to the greedy tokens).
Drafters (no extra model, deterministic):
  ctx      last n tokens (n = 3, 2, 1) matched earlier in the context: propose what followed
  bigram   most frequent next token after the current one, from the training data
  ctx+bi   ctx where it matches, bigram otherwise
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "engine"))
from export import load_float  # noqa: E402
from model import quantize_  # noqa: E402

N_PROMPTS, PROMPT, GEN, DMAX = 40, 24, 64, 8


def bigram_table(vocab):
    tr = np.memmap(ROOT / "data" / "train.bin", dtype=np.uint16, mode="r")[:20_000_000].astype(np.int64)
    counts = np.zeros((vocab, vocab), np.int32)
    np.add.at(counts, (tr[:-1], tr[1:]), 1)
    return counts.argmax(1)


def draft_ctx(seq, d):
    for n in (3, 2, 1):
        if len(seq) <= n:
            continue
        key = seq[-n:]
        for i in range(len(seq) - n - 1, -1, -1):        # most recent earlier match
            if seq[i:i + n] == key:
                return seq[i + n:i + n + d]
    return []


def draft(seq, d, kind, bi):
    out = []
    if kind in ("ctx", "ctx+bi"):
        out = draft_ctx(seq, d)
    if kind in ("bigram", "ctx+bi"):
        cur = list(seq) + list(out)
        while len(out) < d:
            nxt = int(bi[cur[-1]])
            out.append(nxt)
            cur.append(nxt)
    return out[:d]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "combo_s0"
    model, cfg, _ = load_float(name)
    quantize_(model, 4)
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    eot = 0
    starts = [i + 1 for i in np.nonzero(np.array(val[:400_000]) == eot)[0][:N_PROMPTS]]
    seqs = []
    with torch.no_grad():
        for s in starts:
            ids = [int(t) for t in val[s:s + PROMPT]]
            for _ in range(GEN):
                lg, _ = model(torch.tensor(ids[-cfg.seq_len:])[None])
                ids.append(int(lg[0, -1].argmax()))
            seqs.append(ids)
    bi = bigram_table(cfg.vocab)
    print(f"{name}: {len(seqs)} prompts x {GEN} greedy tokens")
    print(f"{'drafter':8s} " + " ".join(f"D={d:<5d}" for d in range(1, DMAX + 1)) + "   (tokens per weight pass)")
    for kind in ("ctx", "bigram", "ctx+bi"):
        row = []
        for D in range(1, DMAX + 1):
            passes = toks = 0
            for ids in seqs:
                pos = PROMPT
                while pos < len(ids):
                    dr = draft(ids[:pos], D, kind, bi)
                    acc = 0
                    while acc < len(dr) and pos + acc < len(ids) and dr[acc] == ids[pos + acc]:
                        acc += 1
                    step = min(1 + acc, len(ids) - pos)
                    toks += step
                    pos += step
                    passes += 1
            row.append(toks / passes)
        print(f"{kind:8s} " + " ".join(f"{v:<7.2f}" for v in row))


if __name__ == "__main__":
    main()
