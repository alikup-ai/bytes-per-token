#!/bin/sh
# Phase 2b arms on the H2 dense shape (6 x 128, ffn 384), compared with
# dense_s$S from sweep_h2.sh. Seed from $1.
#   relu / relu2 : sparse FFN gate, so dead neurons can be skipped (H6)
#   qat4 / qat2  : weights fake-quantized during training (group 64)
# Arms with a final.json are skipped; killed arms resume from ckpt.pt.
set -e
cd "$(dirname "$0")"
P=../.venv/Scripts/python
S=${1:-0}
T=20e6
run() {
    if [ -f "../runs/$1/final.json" ]; then echo "skip $1"; return; fi
    n=$1; shift
    $P train.py --name "$n" --seed $S --tokens $T "$@"
}
run relu_s$S  --act relu
run relu2_s$S --act relu2
run qat4_s$S  --qbits 4
run qat2_s$S  --qbits 2
