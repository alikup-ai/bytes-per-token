"""Teach a trained story model to answer questions about a story.

    .venv/Scripts/python train/finetune_qa.py --base combo_s0 --name combo_qa_s0 --steps 2000
    .venv/Scripts/python train/finetune_qa.py --base combo_s0 --eval-only     # the base model, same test

Starts from runs/<base>/model.pt (same architecture and QAT setting). Each
batch holds QA rows from train/qa_data.py (loss on the answer tokens only) and
plain story windows (full loss, so the model keeps writing stories).

Reports, on held-out stories: loss on answer tokens, exact match of the greedy
answer against the rule-made gold answer, per question type, and the story
validation loss (to check nothing was forgotten).
Writes runs/<name>/model.pt, log.jsonl, final.json, samples.txt.
"""
import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from model import LM, Config
from train import evaluate, keep_awake

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
ROW = 257


def load_rows(name):
    x = np.fromfile(DATA / f"qa_{name}.bin", dtype=np.uint16).reshape(-1, ROW)
    m = np.fromfile(DATA / f"qa_{name}.mask.bin", dtype=np.uint8).reshape(-1, ROW)
    return x, m


def qa_loss(model, x, m):
    logits, _ = model(x[:, :-1])
    y = x[:, 1:].clone()
    y[m[:, 1:] == 0] = -100
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)


@torch.no_grad()
def answer(model, tok, prompt, max_new=40):
    ids = tok.encode(prompt).ids
    eot = tok.token_to_id("<|endoftext|>")
    out = []
    for _ in range(max_new):
        x = torch.tensor([ids[-256:]])
        logits, _ = model(x)
        t = int(logits[0, -1].argmax())
        if t == eot:
            break
        ids.append(t)
        out.append(t)
        if "\n" in tok.decode(out):
            break
    return tok.decode(out).split("\n")[0].strip()


@torch.no_grad()
def exact_match(model, tok, n, samples=None):
    model.eval()
    rows = [json.loads(l) for l in open(DATA / "qa_val.jsonl", encoding="utf-8")]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), size=min(n, len(rows)), replace=False)
    hit, per = 0, defaultdict(lambda: [0, 0])
    for k, i in enumerate(pick):
        r = rows[i]
        a = answer(model, tok, r["prompt"])
        ok = a == r["answer"]
        hit += ok
        kind = " ".join(r["question"].split()[:2]) if not r["question"].startswith("What did") else \
            "What did X " + r["question"].split()[3].rstrip("?")
        kind = kind if kind.startswith("What did") is False else ("What did X say" if kind.endswith("say") else "What did X <verb>")
        per[kind][0] += ok
        per[kind][1] += 1
        if samples is not None and k < 12:
            samples.write(f"{r['prompt'][-300:]}\n  model: {a}\n  gold:  {r['answer']}\n\n")
    model.train()
    return hit / len(pick), {k: round(v[0] / v[1], 3) for k, v in sorted(per.items())}


def main():
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="combo_s0")
    ap.add_argument("--name", default="combo_qa_s0")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--story-frac", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--em", type=int, default=600, help="held-out questions for exact match")
    ap.add_argument("--eval-only", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    base = json.loads((ROOT / "runs" / a.base / "final.json").read_text())
    cfg = Config(**base["cfg"])
    model = LM(cfg)
    model.load_state_dict(torch.load(ROOT / "runs" / a.base / "model.pt", weights_only=True))
    tok = Tokenizer.from_file(str(DATA / "tok4096.json"))
    val_story = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    xv, mv = load_rows("val")
    xv_t, mv_t = torch.from_numpy(xv[:2000].astype(np.int64)), torch.from_numpy(mv[:2000])

    def report(step, t0, samples=None):
        model.eval()
        with torch.no_grad():
            al = sum(qa_loss(model, xv_t[i:i + 50], mv_t[i:i + 50]).item() for i in range(0, 2000, 50)) / 40
        sl = evaluate(model, val_story, cfg.seq_len, int(2e5))
        em, per = exact_match(model, tok, a.em if samples is not None else 200, samples)
        rec = dict(step=step, answer_loss=round(al, 4), story_val_loss=round(sl, 4), exact_match=round(em, 3),
                   per_type=per, elapsed_s=round(time.time() - t0, 1))
        print(json.dumps(rec), flush=True)
        return rec

    out = ROOT / "runs" / (a.base + "_zeroshot" if a.eval_only else a.name)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if a.eval_only:
        with open(out / "samples.txt", "w", encoding="utf-8") as smp:
            rec = report(0, t0, smp)
        (out / "final.json").write_text(json.dumps(dict(base=a.base, **rec), indent=2))
        return

    xt, mt = load_rows("train")
    story = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    gen = torch.Generator().manual_seed(1000 + a.seed)
    n_story = int(round(a.batch * a.story_frac))
    n_qa = a.batch - n_story
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = 50
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / a.steps))))
    log = open(out / "log.jsonl", "w")
    # step 0 is the base model: measured once with --eval-only (runs/<base>_zeroshot)
    T = cfg.seq_len
    for step in range(a.steps):
        qi = torch.randint(len(xt), (n_qa,), generator=gen).numpy()
        x = torch.from_numpy(xt[qi].astype(np.int64))
        m = torch.from_numpy(mt[qi])
        loss = qa_loss(model, x, m)
        if n_story:
            si = torch.randint(len(story) - T - 1, (n_story,), generator=gen)
            xs = torch.stack([torch.from_numpy(story[i:i + T + 1].astype(np.int64)) for i in si])
            _, ls = model(xs[:, :-1].contiguous(), xs[:, 1:].contiguous())
            loss = (loss * n_qa + ls * n_story) / a.batch
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % 500 == 0 and step + 1 < a.steps:
            log.write(json.dumps(report(step + 1, t0)) + "\n"); log.flush()
    with open(out / "samples.txt", "w", encoding="utf-8") as smp:
        rec = report(a.steps, t0, smp)
    log.write(json.dumps(rec) + "\n")
    torch.save(model.state_dict(), out / "model.pt")
    (out / "final.json").write_text(json.dumps(dict(base=a.base, cfg=cfg.to_dict(), steps=a.steps, lr=a.lr,
                                                     story_frac=a.story_frac, final_val_loss=rec["story_val_loss"],
                                                     **rec), indent=2))


if __name__ == "__main__":
    main()
