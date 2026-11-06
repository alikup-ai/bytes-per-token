"""Architectures and the per-token work they imply.

`ops(arch, engine, pos)` walks one decode step the way llm_forward does in
refs/esp32-ai/runtime/llm.h (and the tinyLLM C reference) and returns a flat
list of ops, each tagged with its profiler stage. Nothing here is fitted: bytes,
MACs and call counts follow from the shapes and the engine's placement rules.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Arch:
    vin: int      # input vocabulary (embedding / PLE table rows)
    vout: int     # rows the output head actually computes
    d: int
    layers: int
    heads: int
    ffn: int
    ple: int      # per-layer embedding dim P (0 = no PLE)
    experts: int = 1  # top-1 MoE FFN: one expert read per token, plus the router


ARCHS = {
    # 28.9M PLE TinyLM; head capped to the 25,353 trained rows (VOCAB_N).
    "tinystories": Arch(vin=32768, vout=25353, d=96, layers=6, heads=4, ffn=66, ple=128),
    # 8.9M Barista: separate 854-class output head.
    "barista": Arch(vin=8057, vout=854, d=128, layers=6, heads=4, ffn=384, ple=128),
}


@dataclass
class Op:
    stage: str
    kind: str            # matvec | lookup | attn | elem
    macs: float = 0.0    # weight MACs (matvec) or fp32 MACs (attn)
    bytes: float = 0.0   # bytes streamed from `tier`
    tier: str = "sram"
    cores: int = 1
    calls: int = 0       # matvec calls (fixed per-call cost)
    transc: float = 0.0  # exp/erf/sin/cos/pow evaluations
    rows: int = 0        # random flash row reads (lookup)


def _mv(stage, rows, cols, eng, *, head=False):
    w = eng["head"] if head else eng["core"]
    # one scale per `group` weights of a row; no "group" (or 0) = one scale per row
    groups = -(-cols // w["group"]) if w.get("group") else 1
    wbytes = rows * cols * w["bits"] / 8 + rows * groups * w["scale_bytes"]
    cores = 2 if (eng["dual_all"] or rows >= eng["dual_min_rows"]) else 1
    return Op(stage, "matvec", macs=rows * cols, bytes=wbytes, tier=w["tier"],
              cores=cores, calls=1)


def ops(a: Arch, eng: dict, pos: int):
    D, L, H, F, P = a.d, a.layers, a.heads, a.ffn, a.ple
    dh = D // H
    out = []
    # input: embedding row (+ PLE table row) from flash, ple_model_proj, RoPE tables
    out.append(Op("input", "lookup", bytes=D / 2 + L * P / 2, tier="flash", rows=2 if P else 1))
    if P:
        out.append(_mv("input", L * P, D, eng))
    out.append(Op("input", "elem", transc=3 * dh / 2))
    kv = eng["kv_bytes"]
    for _ in range(L):
        out.append(_mv("attn", 3 * D, D, eng))
        n = pos + 1
        out.append(Op("attn", "attn", macs=2 * n * D, bytes=2 * n * D * kv,
                      tier=eng["kv_tier"], cores=eng["attn_cores"], transc=H * n))
        out.append(_mv("attn", D, D, eng))
        if a.experts > 1:
            out.append(_mv("ffn", a.experts, D, eng))   # router; experts live wherever `core` does
        out.append(_mv("ffn", F, D, eng))
        out.append(_mv("ffn", F, D, eng))
        out.append(Op("ffn", "elem", transc=F))
        out.append(_mv("ffn", D, F, eng))
        if P:
            out.append(_mv("ple", P, D, eng))
            out.append(Op("ple", "elem", transc=P))
            out.append(_mv("ple", D, P, eng))
    out.append(_mv("head", a.vout, D, eng, head=True))
    return out
