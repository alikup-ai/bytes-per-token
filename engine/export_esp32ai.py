"""Convert slvDev/esp32-ai's released TinyStories model (MIT) to our engine's pack,
exactly: same int4 codes, same scales, no requantization.

    .venv/Scripts/python engine/export_esp32ai.py            # downloads, verifies, converts

Writes engine/out/esp32ai_ts.pack (+ .ref.json, .torch_logits.npy for engine/test.py).

Their format (refs/esp32-ai/runtime/llm.h, llm_load): 56-byte header, then tensors
in a fixed order. A quant tensor is int32 group, codes (row-major, 2 per byte,
low nibble = even column, code = value + 8) and fp16 scales per (row, group of
`group` columns). Weights are integer levels: w = (code - 8) * scale.

Mapping to our pack v3 (lvl 0 = integer levels):
  * codes repacked into PIE blocks; columns padded to a multiple of 32 with code 8 (= 0);
  * ffn 66 padded to 96 (zero gate/up rows, zero down columns: silu(0) * 0 = 0);
  * scales: one group per row for every matvec tensor (their group 128 >= row
    length), fp16 -> fp32 exactly; the PLE table (row lookups only) keeps its
    fp16 scales per 128 columns;
  * one embedding copy, rows reordered so the head (first 25,353 tokens) is
    k-means clustered for the two-stage head; emb_row maps token -> row.
The PyTorch reference is their own TinyLM (refs/esp32-ai/src/model.py) with the
dequantized weights, so engine-vs-reference differences are only our int8
activations and the two-stage head.
"""
import hashlib
import json
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "engine" / "out"
SRC = OUT / "esp32ai"
sys.path.insert(0, str(ROOT / "engine"))
from export import PROMPTS, align, kmeans  # noqa: E402  (imports our train/model.py as `model`)

import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("esp32ai_model", ROOT / "refs" / "esp32-ai" / "src" / "model.py")
_their = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_their)
TConfig, TinyLM = _their.Config, _their.TinyLM                 # their model, under its own name

REPO = "https://huggingface.co/slvDev/esp32-ai-tinystories/resolve/main/"
PINNED = {  # refs/esp32-ai/scripts/fetch_model.sh
    "model.bin": ("1d8326c05c383ccfa615f5455575802817cb453dbc7ab28875d41a9dbb45477e", 14912348),
    "tokenizer.json": ("4e28163669f2249af31a528a54fc25064dcbd0a34edbfa7bedb16d2d600ec7ae", 1788896),
}
CLUSTERS, GEN, EVAL = 256, 48, 512


def fetch():
    SRC.mkdir(parents=True, exist_ok=True)
    for name, (sha, size) in PINNED.items():
        p = SRC / name
        if not p.exists():
            print(f"downloading {name}")
            urllib.request.urlretrieve(REPO + name, p)
        data = p.read_bytes()
        assert len(data) == size and hashlib.sha256(data).hexdigest() == sha, f"{name}: not the pinned release"


def parse(blob):
    magic, version, hbytes, flags = struct.unpack_from("<4I", blob, 0)
    assert magic == 0x00454C50 and version == 1, (hex(magic), version)
    vin, vout = struct.unpack_from("<2I", blob, 16)
    dim, L, H, Fh, P, T, group = struct.unpack_from("<7i", blob, 24)
    theta = struct.unpack_from("<f", blob, 52)[0]
    cfg = dict(vin=vin, vout=vout, d=dim, layers=L, heads=H, ffn=Fh, ple=P, ctx=T, group=group, theta=theta,
               tied=bool(flags & 1))
    off = hbytes

    def q(rows, cols):
        nonlocal off
        g = struct.unpack_from("<i", blob, off)[0]; off += 4
        rb, ng = (cols + 1) // 2, (cols + g - 1) // g
        raw = np.frombuffer(blob, np.uint8, rows * rb, off).reshape(rows, rb); off += rows * rb
        sc = np.frombuffer(blob, "<f2", rows * ng, off).reshape(rows, ng); off += rows * ng * 2
        codes = np.empty((rows, rb * 2), np.uint8)
        codes[:, 0::2], codes[:, 1::2] = raw & 15, raw >> 4
        return dict(codes=codes[:, :cols].copy(), scales=sc.copy(), group=g)

    def f(n):
        nonlocal off
        a = np.frombuffer(blob, "<f4", n, off).copy(); off += 4 * n
        return a

    t = {"tok_emb": q(vin, dim), "ple_model_proj": q(L * P, dim), "ple_proj_norm": f(P), "ple_table": q(vin, L * P)}
    for i in range(L):
        t[f"{i}.attn_norm"] = f(dim); t[f"{i}.qkv"] = q(3 * dim, dim); t[f"{i}.attn_proj"] = q(dim, dim)
        t[f"{i}.ffn_norm"] = f(dim); t[f"{i}.gate"] = q(Fh, dim); t[f"{i}.up"] = q(Fh, dim); t[f"{i}.down"] = q(dim, Fh)
        t[f"{i}.ple_gate"] = q(P, dim); t[f"{i}.ple_proj"] = q(dim, P); t[f"{i}.ple_norm"] = f(dim)
    t["out_norm"] = f(dim)
    assert cfg["tied"] and off <= len(blob)
    return cfg, t


def deq(qt):
    c = qt["codes"].astype(np.float32) - 8.0
    s = np.repeat(qt["scales"].astype(np.float32), qt["group"], axis=1)[:, :c.shape[1]]
    return c * s


def pad_q(qt, rows=None, cols=None):
    """Pad with code 8 (value 0); scales of padded rows are 0."""
    c, s = qt["codes"], qt["scales"]
    r0, c0 = c.shape
    rows, cols = rows or r0, cols or c0
    cc = np.full((rows, cols), 8, np.uint8); cc[:r0, :c0] = c
    ss = np.zeros((rows, s.shape[1]), np.float16); ss[:r0] = s
    return dict(codes=cc, scales=ss, group=qt["group"])


def pack_codes(codes):
    rows, cols = codes.shape
    assert cols % 32 == 0
    q = codes.reshape(-1, 2, 16).astype(np.uint8)
    return (q[:, 0] | (q[:, 1] << 4)).reshape(rows, cols // 2).tobytes()


def ceil32(n):
    return (n + 31) // 32 * 32


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--their-bench", action="store_true",
                    help="prompt 0 = their 'Once upon a time' and 200 generated tokens (positions 4..203, as their firmware)")
    a = ap.parse_args()
    global GEN
    prompts_txt = PROMPTS
    if a.their_bench:
        GEN, prompts_txt = 200, ["Once upon a time"] + PROMPTS
    fetch()
    blob = (SRC / "model.bin").read_bytes()
    cfg, t = parse(blob)
    d, L, P, Fh, V, Vo = cfg["d"], cfg["layers"], cfg["ple"], cfg["ffn"], cfg["vin"], cfg["vout"]
    Fp = ceil32(Fh)
    print(f"esp32-ai TinyStories: vocab {V} (head {Vo}), d {d}, layers {L}, heads {cfg['heads']}, "
          f"ffn {Fh} -> {Fp}, ple {P}, group {t['tok_emb']['group']}")

    # ---- reference: their own model, dequantized weights
    tc = TConfig(arm="ple", vocab_size=V, out_vocab_size=Vo, d_model=d, n_layers=L, n_heads=cfg["heads"],
                 ffn_hidden=Fh, seq_len=cfg["ctx"], ple_dim=P, rope_theta=cfg["theta"])
    ref = TinyLM(tc).eval()
    W = lambda k: torch.from_numpy(deq(t[k]))  # noqa: E731
    with torch.no_grad():
        ref.tok_emb.weight.copy_(W("tok_emb"))
        ref.head.weight.copy_(W("tok_emb")[:Vo])         # tied: first vout rows
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

    # ---- single embedding copy, head rows clustered
    emb = torch.from_numpy(deq(t["tok_emb"]))
    cent, a = kmeans(emb[:Vo], CLUSTERS)
    order = torch.argsort(a, stable=True).numpy()
    rows = np.concatenate([order, np.arange(Vo, V)])            # new row -> token
    emb_row = np.empty(V, np.uint32); emb_row[rows] = np.arange(V)
    head_off = np.concatenate([[0], np.cumsum(np.bincount(a.numpy(), minlength=CLUSTERS))]).astype("<u4")
    ce = cent.numpy()                                             # centroids, integer levels, one scale per row
    cs = np.abs(ce).max(1, keepdims=True).clip(1e-8) / 7.0
    cq = dict(codes=(np.clip(np.round(ce / cs), -7, 7) + 8).astype(np.uint8), scales=cs.astype(np.float16), group=d)
    emb_q = dict(codes=t["tok_emb"]["codes"][rows], scales=t["tok_emb"]["scales"][rows], group=t["tok_emb"]["group"])

    # ---- tensors in engine order: (name, kind, rows, cols, gl, codes, scales) ; kind 0 fp32 scales, 5 fp16
    items = []

    def addq(name, qt, cols_pad=None, rows_pad=None, fp16=False):
        qt = pad_q(qt, rows=rows_pad, cols=cols_pad or ceil32(qt["codes"].shape[1]))
        r, c = qt["codes"].shape
        gl = c if qt["group"] >= qt["codes"].shape[1] else qt["group"]   # one group per row unless the row is longer
        assert c % gl == 0 and gl % 32 == 0, (name, c, gl)
        sc = qt["scales"][:, : c // gl]
        items.append((name, 5 if fp16 else 0, r, c, gl, pack_codes(qt["codes"]),
                      sc.astype("<f2" if fp16 else "<f4").tobytes()))

    def addf(name, arr):
        arr = np.asarray(arr, "<f4")
        r, c = (arr.shape if arr.ndim == 2 else (1, arr.shape[0]))
        items.append((name, 1, r, c, 0, arr.tobytes(), None))

    def addu(name, arr):
        items.append((name, 2, 1, len(arr), 0, np.asarray(arr, "<u4").tobytes(), None))

    addq("emb", emb_q)
    addf("norm", t["out_norm"]); addf("rope_cos", ref.cos.numpy()); addf("rope_sin", ref.sin.numpy())
    addq("ple_table", t["ple_table"], fp16=True)
    addq("ple_mproj", t["ple_model_proj"]); addf("ple_pnorm", t["ple_proj_norm"])
    for i in range(L):
        addf(f"L{i}.n1", t[f"{i}.attn_norm"]); addq(f"L{i}.qkv", t[f"{i}.qkv"]); addq(f"L{i}.proj", t[f"{i}.attn_proj"])
        addf(f"L{i}.n2", t[f"{i}.ffn_norm"])
        addq(f"L{i}.gate", t[f"{i}.gate"], rows_pad=Fp); addq(f"L{i}.up", t[f"{i}.up"], rows_pad=Fp)
        addq(f"L{i}.down", t[f"{i}.down"], cols_pad=Fp)
        addq(f"L{i}.ple_gate", t[f"{i}.ple_gate"]); addq(f"L{i}.ple_proj", t[f"{i}.ple_proj"])
        addf(f"L{i}.ple_norm", t[f"{i}.ple_norm"])
    addq("head_cent", cq); addu("head_perm", rows[:Vo]); addu("head_off", head_off); addu("emb_row", emb_row)

    # ---- layout
    head_fmt, dir_fmt = "<8s" + "I" * 15, "<24s6I"
    off = len(align(b"\0" * struct.calcsize(head_fmt)))
    dir_off = off
    off += len(align(b"\0" * struct.calcsize(dir_fmt) * len(items)))
    data, entries = b"", []
    for name, kind, r, c, gl, wb, sb in items:
        ow = off + len(data); data += align(wb)
        os_ = 0
        if sb is not None:
            os_ = off + len(data); data += align(sb)
        entries.append(struct.pack(dir_fmt, name.encode(), kind, r, c, ow, os_, gl))
    vocab_off = off + len(data)
    vocab = align(np.zeros(V + 1, "<u4").tobytes())                # token strings not needed on the board
    test_off = vocab_off + len(vocab)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(SRC / "tokenizer.json"))
    prompts = [tok.encode(p).ids for p in prompts_txt]
    text = (ROOT / "data" / "valid.txt").read_text(encoding="utf-8", errors="ignore")[:20000]
    ev = np.array(tok.encode(text).ids[:EVAL], dtype="<u2")
    assert len(ev) == EVAL and ev.max() < Vo
    tests = struct.pack("<3I", len(prompts), GEN, EVAL)
    for p in prompts:
        tests += struct.pack("<I", len(p)) + np.array(p, dtype="<u2").tobytes()
    tests = align(tests + ev.tobytes())
    total = test_off + len(tests)
    header = struct.pack(head_fmt, b"MCULM03\0", V, d, L, cfg["heads"], Fp, P, 0, cfg["ctx"], len(items),
                         dir_off, vocab_off, test_off, total, Vo, 0)
    pack = align(header) + align(b"".join(entries)) + data + vocab + tests
    assert len(pack) == total
    name = "esp32ai_ts"
    (OUT / f"{name}.pack").write_bytes(pack)

    # ---- PyTorch reference results
    with torch.no_grad():
        nll, argmax = [], []
        T = cfg["ctx"]
        for k in range(0, EVAL, T):
            x = torch.from_numpy(ev[k:k + T].astype(np.int64))[None]
            logits, _ = ref(x)
            lp = F.log_softmax(logits[0, :-1], -1)
            nll += (-lp.gather(1, x[0, 1:, None])[:, 0]).tolist()
            argmax += logits[0].argmax(-1).tolist()
            if k == 0:
                np.save(OUT / f"{name}.torch_logits.npy", logits[0].numpy().astype(np.float32))
        gens, margins = [], []
        for p in prompts:
            ids, mg = list(p), []
            for _ in range(GEN):
                lg, _ = ref(torch.tensor(ids)[None])
                top = lg[0, -1].topk(2).values
                mg.append(float(top[0] - top[1]))
                ids.append(int(lg[0, -1].argmax()))
            gens.append(ids[len(p):]); margins.append(mg)
    refj = dict(name=name, trained_bits=4, cfg=dict(vocab=Vo, vin=V, d=d, layers=L, ffn=Fh, ple=P),
                eval_tokens=EVAL, torch_q4_nll=float(np.mean(nll)), torch_argmax=argmax, torch_gen=gens,
                torch_gen_margin=margins, prompts=prompts_txt, prompt_ids=prompts, pack_bytes=total)
    (OUT / f"{name}.ref.json").write_text(json.dumps(refj))
    print(f"{name}: pack {total / 1e6:.2f} MB, {len(items)} tensors, reference NLL {np.mean(nll):.4f} "
          f"on {len(nll)} targets")
    print("sample:", repr(prompts_txt[0] + tok.decode(gens[0]))[:200])


if __name__ == "__main__":
    main()
