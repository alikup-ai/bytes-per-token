"""Can a ReLU arm skip its dead FFN neurons at run time? (phase 2b, H6)

    .venv/Scripts/python train/predictor_2b.py relu_s0

Two ways to know which neurons are live before reading their weights:
  gate-first  compute the whole gate matrix (1/3 of FFN bytes), then read the
              up row and down column only where relu(gate) > 0. Exact: no loss.
  predictor   a low-rank D -> r -> F classifier per layer (Deja Vu style),
              trained here on the frozen model's own activations. Reads the
              predictor plus gate/up/down only for predicted-live neurons;
              misses change the output, so val loss is measured with it on.
FFN bytes read are reported relative to the dense FFN (3*D*F weights per layer).
"""
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import eval_2b as E
from model import FFN

R, TRAIN_TOKENS, EPOCHS = 32, 400_000, 4
THRESHOLDS = (0.5, 0.3, 0.2, 0.1)


@torch.no_grad()
def collect(model, arr, n_tokens, seed=0):
    """(ffn input, live mask) pairs per layer from training text."""
    ffns = [b.ffn for b in model.blocks]
    xs = [[] for _ in ffns]
    ys = [[] for _ in ffns]
    def hook(k):
        def fn(m, i, o):   # must return None, or it replaces the module output
            xs[k].append(i[0].reshape(-1, i[0].shape[-1]))
            ys[k].append((m.gate(i[0]) > 0).reshape(-1, m.gate.out_features))
        return fn

    hooks = [f.register_forward_hook(hook(k)) for k, f in enumerate(ffns)]
    g = torch.Generator().manual_seed(seed)
    T = model.c.seq_len
    for _ in range(n_tokens // (32 * T)):
        ix = torch.randint(len(arr) - T - 1, (32,), generator=g)
        model.hidden(torch.stack([torch.from_numpy(arr[i:i + T].astype(np.int64)) for i in ix]))
    for h in hooks:
        h.remove()
    return [(torch.cat(a), torch.cat(b).float()) for a, b in zip(xs, ys)]


def train_predictor(x, y, d, f):
    p = nn.Sequential(nn.Linear(d, R, bias=False), nn.Linear(R, f))
    opt = torch.optim.Adam(p.parameters(), lr=3e-3)
    pos = ((1 - y.mean()) / y.mean().clamp_min(1e-3)).clamp(max=20)  # favour recall
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(x)).split(4096):
            loss = F.binary_cross_entropy_with_logits(p(x[idx]), y[idx], pos_weight=pos)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return p.eval()


def gated_forward(ffn, pred, thr):
    def fwd(x):
        g = F.relu(ffn.gate(x))
        keep = torch.sigmoid(pred(x)) > thr
        ffn.stats[0] += keep.float().mean().item()
        live = g > 0
        ffn.stats[1] += (keep & live).float().sum().item() / max(live.float().sum().item(), 1)
        ffn.stats[2] += 1
        return ffn.down(g * keep * ffn.up(x))
    return fwd


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "relu_s0"
    model, cfg = E.load(name)
    for p_ in model.parameters():
        p_.requires_grad_(False)
    train = np.memmap(E.ROOT / "data" / "train.bin", dtype=np.uint16, mode="r")
    val = np.memmap(E.ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    d, f, L = cfg.d, cfg.ffn, cfg.layers
    base = E.val_loss(model, val)
    live = E.live_fraction(model, val)
    mean_live = sum(live) / L
    print(f"== {name}: val {base:.4f} (200k tokens), live neurons {mean_live:.1%}")
    print(f"   gate-first (exact): FFN bytes read {1 / 3 + 2 / 3 * mean_live:.1%}, loss unchanged")

    data = collect(model, train, TRAIN_TOKENS)
    preds = []
    for k, (x, y) in enumerate(data):
        with torch.enable_grad():
            preds.append(train_predictor(x, y, d, f))
    pred_frac = (d * R + R * f) / (3 * d * f)
    print(f"   predictor r={R}: its own size = {pred_frac:.1%} of FFN bytes")
    blocks = [b.ffn for b in model.blocks]
    for thr in THRESHOLDS:
        for b, p in zip(blocks, preds):
            b.stats = [0.0, 0.0, 0]
            b.forward = gated_forward(b, p, thr)
        loss = E.val_loss(model, val)
        kept = sum(b.stats[0] / b.stats[2] for b in blocks) / L
        rec = sum(b.stats[1] / b.stats[2] for b in blocks) / L
        print(f"   thr {thr:.2f}: neurons read {kept:.1%}, recall of live {rec:.3f}, "
              f"FFN bytes read {pred_frac + kept:.1%}, val {loss:.4f} ({loss - base:+.4f})")
    for b in blocks:
        del b.forward


if __name__ == "__main__":
    main()
