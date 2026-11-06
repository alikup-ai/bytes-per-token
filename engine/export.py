"""Export a trained arm to the engine's binary format, plus test data and
PyTorch reference results.

    .venv/Scripts/python engine/export.py qat4_s0 [--eval-tokens 512]

Writes engine/out/<name>.pack = model blob + test section, and
engine/out/<name>.ref.json (PyTorch results the engine is checked against).

Weights are quantized exactly as train/model.py FakeQuant does (int4, group 64,
fp32 absmax/7.5 scale, levels (q + 0.5) * s with q in -8..7), so a QAT model
runs as trained and a float model gets post-training quantization.

Blob layout (little endian, every section 64-byte aligned):
  header   "MCULM01\\0", u32 vocab d layers heads ffn ple act ctx n_tensors,
           u32 dir_off vocab_off test_off total
  dir      n_tensors x {char name[24]; u32 kind(0=q4,1=f32) rows cols off_w off_s}
  q4 data  rows x cols/2 bytes; each 64-weight group = two 16-byte PIE blocks,
           block byte j = w[j] (low nibble, +8 bias) | w[16+j] << 4
  q4 scale rows x cols/64 fp32 (fp16 was tried: slower on the S3, see docs/phase3-engine.md)
  u32 arr  kind 2: rows=1, cols=n, n little-endian u32 (head_perm, head_off)
  q4 codes kind 3: codes only, no scales (L*.downT: the codes of `down` transposed,
           [ffn][d], so one live neuron is one contiguous row; scales stay in `down`)
  vocab    u32 off[vocab+1] then the raw bytes of every token

Two-stage head (--clusters C, default 256; 0 disables): the output rows are
k-means clustered and stored again in cluster order ("head"), with C centroid
rows ("head_cent"), head row -> token id ("head_perm") and cluster row offsets
("head_off"). Rows keep their own int4 groups, so a head row gives exactly the
logit the tied embedding would.
  tests    u32 n_prompts, n_gen, n_eval, then per prompt u32 len + u16 ids,
           then n_eval u16 val tokens
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from data import files  # noqa: E402
from model import LM, Config, quantize_  # noqa: E402

OUT = ROOT / "engine" / "out"
ACTS = {"silu": 0, "relu": 1, "relu2": 2}
PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom had a big red ball. He",
    "The sun was shining and the birds were singing.",
    "One day, a small dog found a",
    "Mom said, \"",
]


def align(b, n=64):
    return b + b"\0" * (-len(b) % n)


def nibbles(w):
    """(int codes 0..15 [rows, cols], fp32 scales [rows, cols/64]) with FakeQuant's exact rounding."""
    w = w.detach().float().contiguous()
    rows, cols = w.shape
    assert cols % 64 == 0, (rows, cols)
    g = w.reshape(-1, 64)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7.5
    q = torch.floor(g / s).clamp(-8, 7).to(torch.int64) + 8          # 0..15
    return q.reshape(rows, cols), s.reshape(rows, cols // 64)


def pack(q):
    """PIE block layout along each row: 32 codes -> 16 bytes, byte j = q[j] | q[16+j] << 4."""
    rows, cols = q.shape
    q = q.reshape(-1, 2, 16)
    return (q[:, 0] | (q[:, 1] << 4)).to(torch.uint8).reshape(rows, cols // 2).numpy().tobytes()


def q4(w):
    """(packed nibbles, fp32 scales)."""
    q, s = nibbles(w)
    return pack(q), s.numpy().astype("<f4").tobytes()


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def tensors(model, cfg):
    """(name, kind, array-or-weight) in the order the engine reads them."""
    t = [("emb", 0, model.emb.weight), ("norm", 1, model.norm.w),
         ("rope_cos", 1, model.cos), ("rope_sin", 1, model.sin)]
    if cfg.ple:
        t += [("ple_table", 0, model.ple_table.weight), ("ple_mproj", 0, model.ple_model_proj.weight),
              ("ple_pnorm", 1, model.ple_proj_norm.w)]
    for i, b in enumerate(model.blocks):
        t += [(f"L{i}.n1", 1, b.n1.w), (f"L{i}.qkv", 0, b.attn.qkv.weight), (f"L{i}.proj", 0, b.attn.proj.weight),
              (f"L{i}.n2", 1, b.n2.w), (f"L{i}.gate", 0, b.ffn.gate.weight), (f"L{i}.up", 0, b.ffn.up.weight),
              (f"L{i}.down", 0, b.ffn.down.weight)]
        if cfg.act != "silu":
            t += [(f"L{i}.downT", 3, b.ffn.down.weight)]
        if cfg.ple:
            t += [(f"L{i}.ple_gate", 0, b.ple_gate.weight), (f"L{i}.ple_proj", 0, b.ple_proj.weight),
                  (f"L{i}.ple_norm", 1, b.ple_norm.w)]
    return t


def kmeans(w, k, iters=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    cent = w[torch.randperm(len(w), generator=g)[:k]].clone()
    for _ in range(iters):
        a = torch.cdist(w, cent).argmin(1)
        for j in range(k):
            sel = w[a == j]
            if len(sel):
                cent[j] = sel.mean(0)
    return cent, torch.cdist(w, cent).argmin(1)


def head_tensors(model, clusters):
    w = model.emb.weight.detach().float()
    cent, a = kmeans(w, clusters)
    perm = torch.argsort(a, stable=True)
    off = torch.cat([torch.zeros(1, dtype=torch.long), torch.bincount(a, minlength=clusters).cumsum(0)])
    return [("head", 0, w[perm]), ("head_cent", 0, cent),
            ("head_perm", 2, perm.numpy().astype("<u4")), ("head_off", 2, off.numpy().astype("<u4"))]


def load_float(name):
    """The arm as trained, with parametrizations removed (plain float weights)."""
    meta = json.loads((ROOT / "runs" / name / "final.json").read_text())
    cfg = Config(**meta["cfg"])
    qbits = cfg.qbits
    model = LM(Config(**{**meta["cfg"], "qbits": 0}))
    sd = torch.load(ROOT / "runs" / name / "model.pt")
    sd = {k.replace("parametrizations.weight.original", "weight"): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return model.eval(), cfg, qbits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--eval-tokens", type=int, default=512)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--clusters", type=int, default=256)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    model, cfg, trained_bits = load_float(a.name)
    assert cfg.experts == 1 and cfg.d % 64 == 0 and cfg.ffn % 64 == 0

    items = tensors(model, cfg) + (head_tensors(model, a.clusters) if a.clusters else [])
    head_fmt = "<8s" + "I" * 13
    dir_fmt = "<24s5I"
    n = len(items)
    off = len(align(b"\0" * struct.calcsize(head_fmt)))
    dir_off = off
    off += len(align(b"\0" * struct.calcsize(dir_fmt) * n))
    data, entries = b"", []
    for name, kind, w in items:
        if kind == 0:
            wb, sb = q4(w)
            ow = off + len(data); data += align(wb)
            os_ = off + len(data); data += align(sb)
            rows, cols = w.shape
        elif kind == 3:
            q, _ = nibbles(w)
            rows, cols = q.shape[1], q.shape[0]
            ow = off + len(data); data += align(pack(q.t().contiguous()))
            os_ = 0
        elif kind == 2:
            rows, cols = 1, len(w)
            ow = off + len(data); data += align(w.tobytes())
            os_ = 0
        else:
            arr = w.detach().float().numpy().astype("<f4")
            rows, cols = (arr.shape if arr.ndim == 2 else (1, arr.shape[0]))
            ow = off + len(data); data += align(arr.tobytes())
            os_ = 0
        entries.append(struct.pack(dir_fmt, name.encode(), kind, rows, cols, ow, os_))
    vocab_off = off + len(data)

    from tokenizers import Tokenizer
    ftok, _, fval = files(cfg.vocab)
    tok = Tokenizer.from_file(str(ftok))
    b2u = bytes_to_unicode()
    raw = [bytes(b2u.get(c, 63) for c in (tok.id_to_token(i) or "")) for i in range(cfg.vocab)]
    offs = np.cumsum([0] + [len(r) for r in raw]).astype("<u4")
    vocab = align(offs.tobytes() + b"".join(raw))
    test_off = vocab_off + len(vocab)

    prompts = [tok.encode(p).ids for p in PROMPTS]
    val = np.memmap(fval, dtype=np.uint16, mode="r")
    ev = np.array(val[:a.eval_tokens], dtype="<u2")
    tests = struct.pack("<3I", len(prompts), a.gen, a.eval_tokens)
    for p in prompts:
        tests += struct.pack("<I", len(p)) + np.array(p, dtype="<u2").tobytes()
    tests = align(tests + ev.tobytes())
    total = test_off + len(tests)

    act = ACTS[cfg.act]
    header = struct.pack(head_fmt, b"MCULM01\0", cfg.vocab, cfg.d, cfg.layers, cfg.heads, cfg.ffn, cfg.ple,
                         act, cfg.seq_len, n, dir_off, vocab_off, test_off, total)
    blob = align(header) + align(b"".join(entries)) + data + vocab + tests
    assert len(blob) == total, (len(blob), total)
    (OUT / f"{a.name}.pack").write_bytes(blob)

    # PyTorch reference: the quantized model (what the engine should approximate).
    quantize_(model, 4)
    with torch.no_grad():
        T = cfg.seq_len
        nll, argmax = [], []
        for k in range(0, a.eval_tokens, T):
            x = torch.from_numpy(ev[k:k + T].astype(np.int64))[None]
            logits, _ = model(x)
            lp = F.log_softmax(logits[0, :-1], -1)
            y = x[0, 1:]
            nll += (-lp.gather(1, y[:, None])[:, 0]).tolist()
            argmax += logits[0].argmax(-1).tolist()
        x0 = torch.from_numpy(ev[:T].astype(np.int64))[None]
        np.save(OUT / f"{a.name}.torch_logits.npy", model(x0)[0][0].numpy().astype(np.float32))
        gens, margins = [], []
        for p in prompts:
            ids, mg = list(p), []
            for _ in range(a.gen):
                lg, _ = model(torch.tensor(ids)[None])
                top = lg[0, -1].topk(2).values
                mg.append(float(top[0] - top[1]))          # how decisive torch's own choice was
                ids.append(int(lg[0, -1].argmax()))
            gens.append(ids[len(p):])
            margins.append(mg)
    ref = dict(name=a.name, trained_bits=trained_bits, cfg=cfg.to_dict(), eval_tokens=a.eval_tokens,
               torch_q4_nll=float(np.mean(nll)), torch_argmax=argmax, torch_gen=gens, torch_gen_margin=margins,
               prompts=PROMPTS, prompt_ids=prompts, pack_bytes=total)
    (OUT / f"{a.name}.ref.json").write_text(json.dumps(ref))
    wbytes = sum(len(q4(w)[0]) + len(q4(w)[1]) for n_, k, w in items if k == 0 and not n_.startswith("head"))
    print(f"{a.name}: pack {total / 1e6:.2f} MB (int4 weights+scales {wbytes / 1e6:.2f} MB), "
          f"{n} tensors, torch int4 NLL {np.mean(nll):.4f} on {len(nll)} targets")


if __name__ == "__main__":
    main()
