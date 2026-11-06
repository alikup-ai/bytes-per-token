"""Our models against esp32-ai's released model on the same text, in bits per byte.

    .venv/Scripts/python train/compare_bpb.py [--stories 5000] runs...

Text: the first N stories of data/valid.txt. Each model tokenizes them with its own
tokenizer (stories joined by <|endoftext|>), is scored on non-overlapping 256-token
windows, and the summed loss is divided by the UTF-8 bytes of the stories. Our QAT
models run with their int4 fake quantization, esp32-ai with its dequantized int4
weights: both as the engine runs them (up to int8 activations).
Caveats: esp32-ai was trained on the first 300 MiB of TinyStories-train.txt, our models on the
first 300 MiB of TinyStoriesV2-GPT4-train.txt (see README.md), so the text is not equally "in distribution" for both;
esp32-ai's head covers only its first 25,353 tokens (targets beyond it are counted).
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from data import EOT, files, stories  # noqa: E402
from model import LM, Config  # noqa: E402

T = 256


def windows_nats(fwd, ids, vout=None):
    """Summed next-token loss over non-overlapping windows; returns (nats, targets, out-of-head targets)."""
    ids = torch.tensor(ids, dtype=torch.long)
    n = (len(ids) - 1) // T
    tot, cnt, miss = 0.0, 0, 0
    with torch.no_grad():
        for k in range(0, n, 16):
            rows = range(k, min(k + 16, n))
            x = torch.stack([ids[r * T:(r + 1) * T] for r in rows])
            y = torch.stack([ids[r * T + 1:(r + 1) * T + 1] for r in rows])
            logits = fwd(x)
            if vout is not None:
                bad = y >= vout
                miss += int(bad.sum())
                y = y.masked_fill(bad, -100)
            tot += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                                   ignore_index=-100, reduction="sum").item()
            cnt += int((y != -100).sum())
    return tot, cnt, miss


def encode(tok, texts):
    eot = tok.token_to_id(EOT)
    ids = []
    for enc in tok.encode_batch(texts):
        ids.extend(enc.ids)
        ids.append(eot)
    return ids


def ours(name, texts):
    m = json.loads((ROOT / "runs" / name / "final.json").read_text())
    cfg = Config(**m["cfg"])
    model = LM(cfg)
    model.load_state_dict(torch.load(ROOT / "runs" / name / "model.pt", weights_only=True))
    model.eval()
    tok = Tokenizer.from_file(str(files(cfg.vocab)[0]))
    return windows_nats(lambda x: model(x)[0], encode(tok, texts)) + (m["stored_params"],)


def esp32ai(texts):
    sys.path.insert(0, str(ROOT / "engine"))
    import export_esp32ai as E
    E.fetch()
    cfg, t = E.parse((E.SRC / "model.bin").read_bytes())
    V, Vo = cfg["vin"], cfg["vout"]
    tc = E.TConfig(arm="ple", vocab_size=V, out_vocab_size=Vo, d_model=cfg["d"], n_layers=cfg["layers"],
                   n_heads=cfg["heads"], ffn_hidden=cfg["ffn"], seq_len=cfg["ctx"], ple_dim=cfg["ple"],
                   rope_theta=cfg["theta"])
    ref = E.TinyLM(tc).eval()
    W = lambda k: torch.from_numpy(E.deq(t[k]))  # noqa: E731
    with torch.no_grad():
        ref.tok_emb.weight.copy_(W("tok_emb"))
        ref.head.weight.copy_(W("tok_emb")[:Vo])
        ref.ple_model_proj.weight.copy_(W("ple_model_proj"))
        ref.ple_proj_norm.weight.copy_(torch.from_numpy(t["ple_proj_norm"]))
        ref.ple_table.weight.copy_(W("ple_table"))
        for i, b in enumerate(ref.blocks):
            b.attn_norm.weight.copy_(torch.from_numpy(t[f"{i}.attn_norm"]))
            b.attn.qkv.weight.copy_(W(f"{i}.qkv")); b.attn.proj.weight.copy_(W(f"{i}.attn_proj"))
            b.ffn_norm.weight.copy_(torch.from_numpy(t[f"{i}.ffn_norm"]))
            b.ffn.gate.weight.copy_(W(f"{i}.gate")); b.ffn.up.weight.copy_(W(f"{i}.up"))
            b.ffn.down.weight.copy_(W(f"{i}.down"))
            b.ple_gate.weight.copy_(W(f"{i}.ple_gate")); b.ple_proj.weight.copy_(W(f"{i}.ple_proj"))
            b.ple_norm.weight.copy_(torch.from_numpy(t[f"{i}.ple_norm"]))
        ref.out_norm.weight.copy_(torch.from_numpy(t["out_norm"]))
    tok = Tokenizer.from_file(str(E.SRC / "tokenizer.json"))
    stored = sum(p.numel() for p in ref.parameters()) - ref.head.weight.numel()   # head tied to tok_emb
    return windows_nats(lambda x: ref(x)[0], encode(tok, texts), vout=Vo) + (stored,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", default=["combo_s0", "comboP128_s0", "varC_s0", "varC_s1"])
    ap.add_argument("--stories", type=int, default=5000)
    a = ap.parse_args()
    texts = [s for s in stories(ROOT / "data" / "valid.txt")[1:] if s][:a.stories]   # [1:]: the file starts mid-story
    nbytes = sum(len(s.encode("utf-8")) for s in texts)
    print(f"{len(texts)} stories, {nbytes / 1e6:.2f} MB of text")
    rows = [(n, *ours(n, texts)) for n in a.runs] + [("esp32-ai TinyStories (their release)", *esp32ai(texts))]
    print(f"{'model':40s} {'stored, M':>9s} {'tokens':>9s} {'bits/byte':>9s} {'nats/token':>10s} {'out-of-head':>11s}")
    for name, nats, cnt, miss, stored in rows:
        print(f"{name:40s} {stored / 1e6:9.1f} {cnt:9d} {nats / math.log(2) / nbytes:9.4f} {nats / cnt:10.4f} {miss:11d}")


if __name__ == "__main__":
    main()
