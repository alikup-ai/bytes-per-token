"""Questions about stories, answered by the board (demo firmware + combo_qa_s0).

    engine/demo/flash_demo.sh COM6 combo_qa_s0
    .venv/Scripts/python engine/qa_test.py --port COM6 [--n 100]

Takes held-out questions from data/qa_val.jsonl (stories the model never saw),
sends "<story>\\nQuestion: <q>\\nAnswer:" with the demo's G command (greedy,
stop at end of text) and compares the board's answer with the gold answer and
with the PyTorch model's answer (train/finetune_qa.py answer()).
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "engine"))
from finetune_qa import answer  # noqa: E402
from model import LM, Config  # noqa: E402
from template_test import Board  # noqa: E402


def kind(q):
    w = q.split()
    if w[0] == "What" and w[1] == "did":
        return "What did X say" if q.endswith("say?") else "What did X <verb>"
    return " ".join(w[:2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM6")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--run", default="combo_qa_s0")
    ap.add_argument("--head", type=float, default=1.0, help="share of head rows read (demo default 0.15; 1 = full head)")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    tok = Tokenizer.from_file(str(ROOT / "data" / "tok4096.json"))
    meta = json.loads((ROOT / "runs" / a.run / "final.json").read_text())
    model = LM(Config(**meta["cfg"]))
    model.load_state_dict(torch.load(ROOT / "runs" / a.run / "model.pt", weights_only=True))
    model.eval()

    rows = [json.loads(l) for l in open(ROOT / "data" / "qa_val.jsonl", encoding="utf-8")]
    b = Board(a.port)
    info = b.ready()
    b.send(f"H {a.head}")
    while not (ln := b.line()).startswith("READY "):
        pass
    info = json.loads(ln[6:])
    # The demo takes at most max_prompt tokens (the page refuses longer stories);
    # longer prompts would lose the question, so they are not asked.
    fits = [i for i, r in enumerate(rows) if len(tok.encode(r["prompt"]).ids) <= info["max_prompt"]]
    print(f"{len(fits)} of {len(rows)} held-out questions fit the board's {info['max_prompt']}-token prompt")
    pick = np.random.default_rng(1).choice(fits, size=a.n, replace=False)
    assert info["vocab"] == 4096, "the board runs another model; flash combo_qa_s0"
    hit, same_torch, per, ms_tok, prefill = 0, 0, defaultdict(lambda: [0, 0]), [], []
    for k, i in enumerate(pick):
        r = rows[i]
        ids = tok.encode(r["prompt"]).ids
        b.send(f"G 40 0 1 1 0 {len(ids)} " + " ".join(map(str, ids)))
        out = []
        while True:
            ln = b.line()
            if ln.startswith("P "):
                prefill.append(float(ln.split()[1]))
            elif ln.startswith("T "):
                t = int(ln.split()[1])
                if t != 0:
                    out.append(t)
            elif ln.startswith("E "):
                n, ms = ln.split()[1:]
                if float(ms) > 0:
                    ms_tok.append(float(ms))
                break
        board = tok.decode(out).split("\n")[0].strip()
        ref = answer(model, tok, r["prompt"])
        ok = board == r["answer"]
        hit += ok
        same_torch += board == ref
        per[kind(r["question"])][0] += ok
        per[kind(r["question"])][1] += 1
        if k < 8 or not ok and k < 40:
            print(f"Q: {r['question']}\n  board: {board}\n  gold:  {r['answer']}" + ("" if board == ref else f"\n  torch: {ref}"))
    print(f"\nboard exact match: {hit}/{a.n} = {hit / a.n:.0%}; board answer == PyTorch answer: {same_torch}/{a.n}")
    for t, (h, n) in sorted(per.items()):
        print(f"  {t:20s} {h}/{n}")
    print(f"story + question read in {np.mean(prefill):.0f} ms (batched), answer at {np.mean(ms_tok):.2f} ms/token"
          f" = {1000 / np.mean(ms_tok):.0f} tok/s; PSRAM {info['psram_mhz']} MHz, head reads {info['head_budget']:.0%} of rows")


if __name__ == "__main__":
    main()
