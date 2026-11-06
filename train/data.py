"""Tokenizer + token files for TinyStories.

A byte-level BPE shared by every arm of one vocabulary size, so validation
loss is comparable across architectures. 4096 by default: small on purpose, the
output head's rows are the vocabulary, and the head is the most expensive stage
on the MCU. Other sizes (8192, 16384) are for the variants in docs/phase2-h2.md;
their losses are compared in bits per byte (`val_bpb`), not per token.

    .venv/Scripts/python train/data.py            # 4096: tok4096.json, train.bin, val.bin
    .venv/Scripts/python train/data.py 8192       # tok8192.json, train_v8192.bin, val_v8192.bin
"""
import math
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

DATA = Path(__file__).resolve().parent.parent / "data"
VOCAB = 4096
EOT = "<|endoftext|>"


def files(vocab=VOCAB):
    """(tokenizer, train tokens, val tokens) for a vocabulary size. 4096 keeps the original names."""
    sfx = "" if vocab == VOCAB else f"_v{vocab}"
    return DATA / f"tok{vocab}.json", DATA / f"train{sfx}.bin", DATA / f"val{sfx}.bin"


def bits_per_byte(mean_nats, val, n_targets, vocab):
    """Mean loss per token (nats) over targets val[1 : n_targets + 1] -> bits per UTF-8 byte of that text.
    End-of-text tokens have no bytes; every tokenizer puts them at the same story boundaries."""
    tok = Tokenizer.from_file(str(files(vocab)[0]))
    nbytes = len(tok.decode(np.asarray(val[1:n_targets + 1]).tolist()).encode("utf-8"))
    return mean_nats * n_targets / math.log(2) / nbytes


def stories(path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = [s.strip() for s in text.split(EOT)]
    return parts[:-1] if path.name.startswith("train_part") else parts  # last may be truncated


def train_tokenizer(train_stories, vocab):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[EOT],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(train_stories[:200_000], trainer=trainer)
    return tok


def encode(tok, items, out):
    eot = tok.token_to_id(EOT)
    ids = []
    for enc in tok.encode_batch([s for s in items if s]):
        ids.extend(enc.ids)
        ids.append(eot)
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(out)
    return len(arr)


def main():
    vocab = int(sys.argv[1]) if len(sys.argv) > 1 else VOCAB
    assert vocab <= 65536, "tokens are stored as uint16"
    ftok, ftrain, fval = files(vocab)
    train = stories(DATA / "train_part.txt")
    valid = stories(DATA / "valid.txt")
    tok = train_tokenizer(train, vocab)
    tok.save(str(ftok))
    n_tr = encode(tok, train, ftrain)
    n_va = encode(tok, valid, fval)
    print(f"vocab {vocab}: stories train {len(train):,} val {len(valid):,}; tokens train {n_tr:,} val {n_va:,}")
    print(f"bytes/token (val): {(DATA / 'valid.txt').stat().st_size / n_va:.2f}")


if __name__ == "__main__":
    main()
