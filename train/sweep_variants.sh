#!/bin/sh
# Model variants from the calibrated cost model (2026-09-29, docs/phase2-h2.md "Variants").
# All are combo-style: ReLU gate, QAT 4-bit, PLE; 20M training tokens each.
#   D : wider trunk (d 160, 4 layers, ffn 384), P 128             vocab 4096
#   B : vocab 8192, P 64 (combo trunk)                               needs data/*_v8192 (train/data.py 8192)
#   C : vocab 8192, P 128, ffn 256                                   needs data/*_v8192
#   G : vocab 16384, P 128, ffn 256                                  needs data/*_v16384
# Losses of different vocabularies are compared in bits per byte (final.json "val_bpb").
set -e
cd "$(dirname "$0")"
P=../.venv/Scripts/python
T=20e6
run() {
    if [ -f "../runs/$1/final.json" ]; then echo "skip $1"; return; fi
    n=$1; shift
    $P train.py --name "$n" --tokens $T --act relu --qbits 4 "$@"
}
for S in 0 1; do run varD_s$S --seed $S --layers 4 --d 160 --heads 5 --ffn 384 --ple 128; done
if [ "$1" = "all" ]; then
    run varB_s0 --seed 0 --vocab 8192 --ffn 320 --ple 64
    for S in 0 1; do run varC_s$S --seed $S --vocab 8192 --ffn 256 --ple 128; done
    run varG_s0 --seed 0 --vocab 16384 --ffn 256 --ple 128
fi
