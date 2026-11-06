"""Per-token latency model for an in-order MCU with blocking memory reads.

Every op costs  compute + bytes / bandwidth  (additive, not max): on the LX7 a
cache miss stalls the core. esp32-tinyLLM measured this directly -- staging the
head as int8 instead of int4 added 1.21 MB/token and cost 14.6 ms against the
13.3 ms the extra bytes take at 91 MB/s (BENCHMARKING.md, head episode 6).

Stage time is linear in five unknown per-engine costs, THETA below. Known
quantities (PSRAM bandwidth, flash row latency) are charged directly.
"""
import numpy as np

from arch import ops

THETA = ["cyc_per_mac", "cyc_per_attn_mac", "cyc_per_transc", "us_per_call", "s_per_flash_byte"]
STAGES = ["input", "attn", "ffn", "ple", "head"]


def _bw(hw, bus, tier, cores):
    key = bus if tier == "psram" else tier
    return hw["bw_Bps"][key][str(cores)]


def op_terms(op, hw, bus):
    """One op as (compute coefficients over THETA, memory coefficients over
    THETA, memory seconds already known). Seconds = c @ theta + m @ theta + k."""
    f = hw["f_hz"]
    c, m, k = np.zeros(len(THETA)), np.zeros(len(THETA)), 0.0
    if op.kind == "matvec":
        c[0] = op.macs / (f * op.cores)
        c[3] = op.calls * 1e-6
    elif op.kind == "attn":
        c[1] = op.macs / (f * op.cores)
        c[2] = op.transc / (f * op.cores)
    elif op.kind == "elem":
        c[2] = op.transc / f
    if op.kind == "lookup":
        k = op.rows * hw["flash_row_us"] * 1e-6
    elif op.tier == "flash":
        m[4] = op.bytes
    elif op.tier == "psram":
        k = op.bytes / _bw(hw, bus, "psram", op.cores)
    return c, m, k


def features(arch, eng, hw, bus, pos):
    """Return (A, b): stage_seconds = A @ theta + b, rows ordered as STAGES."""
    A = np.zeros((len(STAGES), len(THETA)))
    b = np.zeros(len(STAGES))
    for op in ops(arch, eng, pos):
        i = STAGES.index(op.stage)
        c, m, k = op_terms(op, hw, bus)
        A[i] += c + m
        b[i] += k
    return A, b


def run_features(run, archs, engines, hw):
    """Average the linear model over the decode positions a run profiles."""
    arch, eng = archs[run["arch"]], engines[run["engine"]]
    if "mean_pos" in run:
        return features(arch, eng, hw, run["bus"], run["mean_pos"])
    # llm_forward cost is affine in pos, so the mean position gives the mean cost.
    mean_pos = run["prompt_len"] + (run["n_generate"] - 1) / 2
    return features(arch, eng, hw, run["bus"], mean_pos)
