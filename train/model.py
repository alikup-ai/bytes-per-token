"""Model family for the iso-bytes-per-token study (plan phase 2, H2).

One trunk (RMSNorm, split-half RoPE, SwiGLU, tied input/output embedding, the
same op set as refs/esp32-ai/src/model.py), three places to put parameters:

  dense   everything in the trunk; every weight is read for every token
  ple     + Gemma-style per-layer embeddings: a [V, L*P] table read one row
            per token (lives in flash on the MCU), gated into each layer
  moe     FFN replaced by E experts with top-1 routing: E times the FFN
            parameters, one expert read per token

`bytes_per_token` is the quantity the study holds fixed: weight bytes the MCU
must read to produce one token, at a given bit width. It counts only what is
touched -- one embedding row, one PLE row, one expert.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab: int = 4096
    d: int = 128
    layers: int = 6
    heads: int = 4
    ffn: int = 384
    ple: int = 0          # per-layer embedding dim P; 0 disables PLE
    experts: int = 1      # >1 turns every FFN into a top-1 MoE
    seq_len: int = 256
    rope_theta: float = 10000.0
    act: str = "silu"     # FFN gate: silu (SwiGLU) | relu | relu2 (sparse, phase 2b)
    qbits: int = 0        # >0: train with weights fake-quantized to this many bits (QAT)

    def to_dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, n, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(n))
        self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.h, self.dh = c.heads, c.d // c.heads
        self.qkv = nn.Linear(c.d, 3 * c.d, bias=False)
        self.proj = nn.Linear(c.d, c.d, bias=False)

    def forward(self, x, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, D))


ACTS = {"silu": F.silu, "relu": F.relu, "relu2": lambda z: F.relu(z).square()}


class FFN(nn.Module):
    """Gated FFN. With a ReLU gate a neuron whose gate is <= 0 contributes
    exactly nothing, so its gate row, up row and down column need not be read."""

    def __init__(self, d, f, act="silu"):
        super().__init__()
        self.gate = nn.Linear(d, f, bias=False)
        self.up = nn.Linear(d, f, bias=False)
        self.down = nn.Linear(f, d, bias=False)
        self.act = ACTS[act]
        self.active = None   # set to a float tensor to record the fraction of live neurons

    def forward(self, x):
        g = self.act(self.gate(x))
        if self.active is not None:
            self.active += (g != 0).float().mean().detach()
        return self.down(g * self.up(x))


class FakeQuant(nn.Module):
    """Symmetric group-wise weight quantization with a straight-through
    gradient. Levels are half-integers (+-0.5 .. +-(2^(b-1)-0.5)) times a
    per-group absmax scale, so 2 bits gives 4 levels with no wasted code.

    Groups run along each row (the last dim), as the engine stores them. A row
    whose length is not a multiple of `group` ends in a shorter group (zero
    padding does not change absmax). Before 2026-09-29 groups were cut from the
    flattened tensor, which crossed row boundaries when cols % group != 0: that
    only affected qat2w_s0/s1 (down, 560 columns); every other arm has
    cols % 64 == 0, where both layouts are identical."""

    def __init__(self, bits, group=64):
        super().__init__()
        self.bits, self.group = bits, group

    def forward(self, w):
        shape, cols = w.shape, w.shape[-1]
        pad = -cols % self.group
        rows = w.reshape(-1, cols)
        if pad:
            rows = F.pad(rows, (0, pad))
        g = rows.reshape(-1, self.group)
        top = 2 ** (self.bits - 1) - 0.5
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / top
        q = (torch.floor(g / s).clamp(-top - 0.5, top - 0.5) + 0.5) * s
        out = (g + (q - g).detach()).reshape(-1, cols + pad)
        return (out[:, :cols] if pad else out).reshape(shape)


def quantize_(model, bits, group=64):
    """Fake-quantize every Linear and Embedding weight in place (norms stay fp)."""
    from torch.nn.utils import parametrize
    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            parametrize.register_parametrization(m, "weight", FakeQuant(bits, group))
    return model


class MoE(nn.Module):
    """Top-1 (Switch) routing. Aux loss keeps experts balanced; without it a
    top-1 router collapses onto one expert and the extra parameters are dead."""

    def __init__(self, d, f, e):
        super().__init__()
        self.router = nn.Linear(d, e, bias=False)
        self.gate = nn.Parameter(torch.randn(e, d, f) * d ** -0.5)
        self.up = nn.Parameter(torch.randn(e, d, f) * d ** -0.5)
        self.down = nn.Parameter(torch.randn(e, f, d) * f ** -0.5)
        self.e = e
        self.aux = torch.zeros(())

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        probs = self.router(flat).softmax(-1)
        p, idx = probs.max(-1)
        out = torch.zeros_like(flat)
        for j in range(self.e):
            sel = (idx == j).nonzero(as_tuple=True)[0]
            if sel.numel() == 0:
                continue
            h = flat[sel]
            y = (F.silu(h @ self.gate[j]) * (h @ self.up[j])) @ self.down[j]
            out[sel] = y * p[sel, None]
        frac = torch.bincount(idx, minlength=self.e).float() / idx.numel()
        self.aux = self.e * (frac * probs.mean(0)).sum()
        return out.view(B, T, D)


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.d), Attention(c)
        self.n2 = RMSNorm(c.d)
        self.ffn = MoE(c.d, c.ffn, c.experts) if c.experts > 1 else FFN(c.d, c.ffn, c.act)
        if c.ple:
            self.ple_gate = nn.Linear(c.d, c.ple, bias=False)
            self.ple_proj = nn.Linear(c.ple, c.d, bias=False)
            self.ple_norm = RMSNorm(c.d)

    def forward(self, x, cos, sin, ple_l=None):
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.ffn(self.n2(x))
        if ple_l is not None:
            x = x + self.ple_norm(self.ple_proj(F.gelu(self.ple_gate(x)) * ple_l))
        return x


class LM(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.emb = nn.Embedding(c.vocab, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.norm = RMSNorm(c.d)
        if c.ple:
            self.ple_table = nn.Embedding(c.vocab, c.layers * c.ple)
            self.ple_model_proj = nn.Linear(c.d, c.layers * c.ple, bias=False)
            self.ple_proj_norm = RMSNorm(c.ple)
        dh = c.d // c.heads
        inv = c.rope_theta ** (-torch.arange(0, dh // 2).float() * 2 / dh)
        ang = torch.outer(torch.arange(c.seq_len).float(), inv)
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)
        self.apply(self._init)
        if c.qbits:
            quantize_(self, c.qbits)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        c = self.c
        logits = self.hidden(idx) @ self.emb.weight.T
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, c.vocab), targets.view(-1))
        return logits, loss

    def hidden(self, idx):
        """Final normed hidden state: the vector the output head multiplies."""
        c = self.c
        B, T = idx.shape
        x = self.emb(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        ple = None
        if c.ple:
            proj = self.ple_model_proj(x).view(B, T, c.layers, c.ple) / math.sqrt(c.d)
            tab = self.ple_table(idx).view(B, T, c.layers, c.ple) * math.sqrt(c.ple)
            ple = (self.ple_proj_norm(proj) + tab) / math.sqrt(2)
        for i, blk in enumerate(self.blocks):
            x = blk(x, cos, sin, None if ple is None else ple[:, :, i])
        return self.norm(x)

    def aux_loss(self):
        return sum(b.ffn.aux for b in self.blocks if isinstance(b.ffn, MoE))


def param_counts(c: Config):
    """(stored parameters, parameters read per token)."""
    d, L, f, V, P, E = c.d, c.layers, c.ffn, c.vocab, c.ple, c.experts
    attn = 4 * d * d
    ffn_one = 3 * d * f
    router = d * E if E > 1 else 0
    ple_layer = 2 * d * P if P else 0
    stored = V * d + L * (attn + E * ffn_one + router + ple_layer)
    read = V * d + d + L * (attn + ffn_one + router + ple_layer)   # head rows + one embedding row
    if P:
        stored += V * L * P + d * L * P
        read += L * P + d * L * P                                   # one table row + proj
    return stored, read


def bytes_per_token(c: Config, bits=4):
    return param_counts(c)[1] * bits / 8
