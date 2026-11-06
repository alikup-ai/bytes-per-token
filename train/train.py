"""Train one arm on CPU and log validation loss.

    .venv/Scripts/python train/train.py --name dense_s0 --d 128 --ffn 384 --tokens 20e6
    .venv/Scripts/python train/train.py --bench     # measure tokens/s only

Writes runs/<name>/log.jsonl (one line per eval) and runs/<name>/final.json.
Every arm sees the same token stream for a given seed, so loss differences
come from the architecture.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from data import bits_per_byte, files
from model import LM, Config, bytes_per_token, param_counts

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def batches(arr, bs, T, gen):
    while True:
        ix = torch.randint(len(arr) - T - 1, (bs,), generator=gen)
        x = torch.stack([torch.from_numpy(arr[i:i + T].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(arr[i + 1:i + T + 1].astype(np.int64)) for i in ix])
        yield x, y


@torch.no_grad()
def evaluate(model, val, T, n_tokens):
    model.eval()
    # Fixed, contiguous, non-overlapping windows: identical for every arm.
    n = n_tokens // T
    tot = 0.0
    for k in range(0, n, 32):
        rows = range(k, min(k + 32, n))
        x = torch.stack([torch.from_numpy(val[r * T:(r + 1) * T].astype(np.int64)) for r in rows])
        y = torch.stack([torch.from_numpy(val[r * T + 1:(r + 1) * T + 1].astype(np.int64)) for r in rows])
        _, loss = model(x, y)
        tot += loss.item() * len(rows)
    model.train()
    return tot / n


def keep_awake():
    """Ask Windows not to sleep while training runs (released when the process exits)."""
    import os
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # CONTINUOUS | SYSTEM_REQUIRED


def main():
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="smoke")
    for k, v in Config().to_dict().items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    ap.add_argument("--tokens", type=float, default=20e6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aux", type=float, default=0.01)
    ap.add_argument("--eval-tokens", type=float, default=1e6)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    cfg = Config(**{k: getattr(a, k) for k in Config().to_dict()})
    model = LM(cfg)
    stored, read = param_counts(cfg)
    _, ftrain, fval = files(cfg.vocab)          # vocab 4096: train.bin / val.bin
    train = np.memmap(ftrain, dtype=np.uint16, mode="r")
    val = np.memmap(fval, dtype=np.uint16, mode="r")
    T, bs = cfg.seq_len, a.batch
    steps = int(a.tokens // (T * bs))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    warm = max(1, steps // 50)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / steps))))
    gen = torch.Generator().manual_seed(1000 + a.seed)
    it = batches(train, bs, T, gen)

    out = ROOT / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(name=a.name, cfg=cfg.to_dict(), stored_params=stored, read_params=read,
                bytes_per_token_4bit=bytes_per_token(cfg, 4), steps=steps, tokens=steps * T * bs,
                lr=a.lr, seed=a.seed, batch=bs)
    print(json.dumps(meta))

    # Resume from the last eval checkpoint if a previous run of this arm died.
    # The batch generator's state is saved too, so a resumed run sees exactly
    # the token stream an uninterrupted one would.
    ckpt, start, t_prev = out / "ckpt.pt", 0, 0.0
    if ckpt.exists() and not a.bench:
        s = torch.load(ckpt, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        gen.set_state(s["gen"])
        start, t_prev = s["step"], s["elapsed_s"]
        print(f"resumed at step {start}", flush=True)
    log = open(out / "log.jsonl", "a" if start else "w")

    t0 = time.time() - t_prev
    rec = None
    for step in range(start, steps):
        x, y = next(it)
        _, loss = model(x, y)
        total = loss + a.aux * model.aux_loss() if cfg.experts > 1 else loss
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if a.bench and step == 19:
            dt = time.time() - t0
            print(f"bench: {20 * T * bs / dt:,.0f} train tokens/s")
            return
        if (step + 1) % a.eval_every == 0 or step + 1 == steps:
            vl = evaluate(model, val, T, int(2e5) if step + 1 < steps else int(a.eval_tokens))
            rec = dict(step=step + 1, tokens=(step + 1) * T * bs, train_loss=loss.item(),
                       val_loss=vl, elapsed_s=round(time.time() - t0, 1))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                            gen=gen.get_state(), step=step + 1, elapsed_s=rec["elapsed_s"]),
                       ckpt.with_suffix(".tmp"))
            ckpt.with_suffix(".tmp").replace(ckpt)
    if rec is None:   # resumed from the final checkpoint: killed after the last eval, before final.json
        rec = dict(val_loss=evaluate(model, val, T, int(a.eval_tokens)), elapsed_s=t_prev)
    n_targets = int(a.eval_tokens) // T * T
    meta.update(final_val_loss=rec["val_loss"], elapsed_s=rec["elapsed_s"],
                val_bpb=bits_per_byte(rec["val_loss"], val, n_targets, cfg.vocab))
    (out / "final.json").write_text(json.dumps(meta, indent=2))
    torch.save(model.state_dict(), out / "model.pt")
    ckpt.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
