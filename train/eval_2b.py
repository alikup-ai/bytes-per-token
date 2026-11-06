"""Phase 2b checks that need no new training, run on trained arms.

    .venv/Scripts/python train/eval_2b.py dense_s0 ple64_s0 ...

1. Post-training quantization: val loss with every Linear/Embedding weight
   rounded to 4 and 2 bits (group 64, absmax scale kept in fp32: the text
   before 2026-09-29 said fp16, but no fp16 rounding was simulated).
   --clip adds a stronger baseline (MSE-chosen clipping, see ptq_clip_);
   --ptq-only runs just the quantization part.
   Our H2 losses are fp32; this is what the 4-bit cost model rows actually get.
2. Two-stage head (screening): k-means the output rows (tied embedding) into
   C clusters, score clusters by h . centroid, then read rows only of the best
   clusters until a row budget is spent. Reports, per budget:
     recall  - true next token inside the rows read
     top1    - argmax over the rows read equals the full-head argmax
     loss    - rows not read get their cluster's centroid logit (a coarse
               stand-in, so the number stays finite)
   Post-hoc clustering, no training: a lower bound on what a trained
   screen (L2S, arXiv 1810.12406) would do.
3. ReLU arms: fraction of FFN neurons live per token (what an oracle could skip).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LM, Config, FFN, quantize_

ROOT = Path(__file__).resolve().parent.parent
T, N_EVAL = 256, int(2e5)
BUDGETS = (0.05, 0.10, 0.20)
C = 64


def load(name, qbits=0):
    m = json.loads((ROOT / "runs" / name / "final.json").read_text())
    cfg = Config(**m["cfg"])
    model = LM(cfg)
    model.load_state_dict(torch.load(ROOT / "runs" / name / "model.pt"))
    if qbits:
        quantize_(model, qbits)
    return model.eval(), cfg


CLIPS = torch.linspace(0.5, 1.0, 21)


@torch.no_grad()
def ptq_clip_(model, bits, grid="half", group=64):
    """Stronger post-training baseline than absmax rounding (review 2026-09-29):
    per 64-weight group, pick the clipping ratio c (0.5..1 x absmax) that
    minimises the squared weight error.
      grid "half": the engine's levels (k + 0.5) * s, no zero (as FakeQuant)
      grid "int" : symmetric integer levels k * s, k in -(2^(b-1)-1)..2^(b-1)-1 (has a zero;
                   one level fewer, not what the engine runs)
    Scales stay fp32, like the engine."""
    for m in model.modules():
        if not isinstance(m, (nn.Linear, nn.Embedding)):
            continue
        w = m.weight
        g = w.reshape(-1, group)
        amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-8)
        best, err = g.clone(), torch.full_like(amax, float("inf"))
        for c in CLIPS:
            if grid == "half":
                top = 2 ** (bits - 1) - 0.5
                s = amax * c / top
                q = (torch.floor(g / s).clamp(-top - 0.5, top - 0.5) + 0.5) * s
            else:
                top = 2 ** (bits - 1) - 1
                s = amax * c / top
                q = torch.round(g / s).clamp(-top, top) * s
            e = (q - g).square().sum(-1, keepdim=True)
            take = e < err
            err = torch.where(take, e, err)
            best = torch.where(take, q, best)
        w.copy_(best.reshape(w.shape))
    return model


def windows(val):
    n = N_EVAL // T
    for k in range(0, n, 32):
        rows = range(k, min(k + 32, n))
        x = torch.stack([torch.from_numpy(val[r * T:(r + 1) * T].astype(np.int64)) for r in rows])
        y = torch.stack([torch.from_numpy(val[r * T + 1:(r + 1) * T + 1].astype(np.int64)) for r in rows])
        yield x, y


@torch.no_grad()
def val_loss(model, val):
    tot = n = 0
    for x, y in windows(val):
        _, loss = model(x, y)
        tot += loss.item() * x.shape[0]
        n += x.shape[0]
    return tot / n


def kmeans(w, k, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    cent = w[torch.randperm(len(w), generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(w, cent).argmin(1)
        for j in range(k):
            sel = w[a == j]
            if len(sel):
                cent[j] = sel.mean(0)
    return cent, a


@torch.no_grad()
def screen(model, val):
    W = model.emb.weight
    V = len(W)
    cent, assign = kmeans(W, C)
    sizes = torch.bincount(assign, minlength=C)
    out = {b: dict(recall=0.0, top1=0.0, loss=0.0, rows=0.0) for b in BUDGETS}
    n = 0
    for x, y in windows(val):
        h = model.hidden(x).reshape(-1, W.shape[1])
        t = y.reshape(-1)
        full = h @ W.T
        cs = h @ cent.T                                   # cluster scores
        order = cs.argsort(1, descending=True)
        csum = sizes[order].cumsum(1)
        for b in BUDGETS:
            m = (csum - sizes[order]) < b * V              # clusters taken until budget spent
            keep_c = torch.zeros_like(cs, dtype=torch.bool).scatter(1, order, m)
            keep = keep_c[:, assign]                       # [N, V] rows read
            approx = torch.where(keep, full, cs[:, assign])
            r = out[b]
            r["recall"] += keep.gather(1, t[:, None]).float().sum().item()
            r["top1"] += (full.masked_fill(~keep, -1e9).argmax(1) == full.argmax(1)).float().sum().item()
            r["loss"] += F.cross_entropy(approx, t, reduction="sum").item()
            r["rows"] += keep.float().mean(1).sum().item()
        n += len(t)
    return {b: {k: v / n for k, v in r.items()} for b, r in out.items()}, sizes


@torch.no_grad()
def live_fraction(model, val):
    ffns = [b.ffn for b in model.blocks if isinstance(b.ffn, FFN)]
    for f in ffns:
        f.active = torch.zeros(())
    k = 0
    for x, _ in windows(val):
        model.hidden(x)
        k += 1
    fr = [(f.active / k).item() for f in ffns]
    for f in ffns:
        f.active = None
    return fr


def main():
    val = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    for name in (a for a in sys.argv[1:] if not a.startswith("--")):
        model, cfg = load(name)
        print(f"\n== {name}  (val loss on {N_EVAL // 1000}k tokens)")
        base = val_loss(model, val)
        print(f"   fp32 {base:.4f}", end="")
        for bits in (4, 2):
            q, _ = load(name, bits)
            print(f" | ptq{bits} {val_loss(q, val):.4f}", end="", flush=True)
        print()
        if "--ptq-only" in flags or "--clip" in flags:
            for bits in (4, 2):
                for grid in ("half", "int"):
                    q = ptq_clip_(load(name)[0], bits, grid)
                    print(f"   ptq{bits} MSE-clipped, {grid} grid: {val_loss(q, val):.4f}", flush=True)
        if "--ptq-only" in flags:
            continue
        res, sizes = screen(model, val)
        print(f"   head screening, {C} k-means clusters (sizes {sizes.min().item()}..{sizes.max().item()}):")
        for b, r in res.items():
            print(f"     budget {b:.0%}: rows read {r['rows']:.1%}  recall {r['recall']:.3f}  "
                  f"top1 {r['top1']:.3f}  loss {r['loss']:.4f} (+{r['loss'] - base:.4f})")
        if cfg.act != "silu":
            fr = live_fraction(model, val)
            print("   live FFN neurons per layer: " + " ".join(f"{v:.1%}" for v in fr))


if __name__ == "__main__":
    main()
