"""Reproduce the quality numbers on a PC, no board needed.

    pip install torch tokenizers safetensors huggingface_hub
    python bench_pc.py                      # varC_long_s0 on 5,000 validation stories (~2 min on a laptop CPU)
    python bench_pc.py combo_s0 --stories 500

Downloads the weights from https://huggingface.co/alikup-ai/bytes-per-token and the validation text
(TinyStoriesV2-GPT4-valid.txt) from https://huggingface.co/datasets/roneneldan/TinyStories, then scores the
first N stories (skipping the first, which the file starts in the middle of) on non-overlapping 256-token
windows. Stories are joined by <|endoftext|>. The model runs with its int4 fake quantization, as on the board.
Result: summed loss in bits divided by the UTF-8 bytes of the stories (bits per byte, tokenizer-independent).
Expected (5,000 stories): varC_long_s0 0.6253, combo_s0 0.7322.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
from model import LM, Config  # noqa: E402

REPO = "alikup-ai/bytes-per-token"
EOT = "<|endoftext|>"
T = 256


def load(name, local=None):
    get = (lambda f: Path(local) / name / f) if local else (lambda f: hf_hub_download(REPO, f"{name}/{f}"))
    meta = json.loads(Path(get("config.json")).read_text())
    model = LM(Config(**meta["cfg"]))
    model.load_state_dict(load_file(get("model.safetensors")))
    return model.eval(), Tokenizer.from_file(str(get("tokenizer.json"))), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="varC_long_s0")
    ap.add_argument("--stories", type=int, default=5000)
    ap.add_argument("--local", help="folder with <model>/{config.json,model.safetensors,tokenizer.json} instead of the Hub")
    a = ap.parse_args()

    model, tok, meta = load(a.model, a.local)
    valid = hf_hub_download("roneneldan/TinyStories", "TinyStoriesV2-GPT4-valid.txt", repo_type="dataset")
    texts = [s.strip() for s in Path(valid).read_text(encoding="utf-8", errors="ignore").split(EOT)]
    texts = [s for s in texts[1:] if s][:a.stories]
    nbytes = sum(len(s.encode("utf-8")) for s in texts)

    eot = tok.token_to_id(EOT)
    ids = []
    for enc in tok.encode_batch(texts):
        ids += enc.ids + [eot]
    ids = torch.tensor(ids)
    n = (len(ids) - 1) // T
    nats = 0.0
    with torch.no_grad():
        for k in range(0, n, 16):
            rows = range(k, min(k + 16, n))
            x = torch.stack([ids[r * T:(r + 1) * T] for r in rows])
            y = torch.stack([ids[r * T + 1:(r + 1) * T + 1] for r in rows])
            logits = model(x)[0]
            nats += F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum").item()
            print(f"\r{min(k + 16, n)}/{n} windows", end="", file=sys.stderr)
    print(file=sys.stderr)
    print(f"model {a.model}: {meta['stored_params'] / 1e6:.1f}M stored params, "
          f"{meta['bytes_per_token_4bit'] / 1e3:.0f} KB of weights read per token at 4 bits")
    print(f"{len(texts)} stories, {nbytes / 1e6:.2f} MB, {n * T} tokens scored")
    print(f"bits per byte: {nats / math.log(2) / nbytes:.4f}   nats per token: {nats / (n * T):.4f}")


if __name__ == "__main__":
    main()
