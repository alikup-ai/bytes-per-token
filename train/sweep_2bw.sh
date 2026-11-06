#!/bin/sh
# 2-bit at equal bytes: a model with 2x the parameters of dense (d=192, ffn 560,
# 3.61M read) trained at 2 bits reads what qat4 (1.80M at 4 bits) reads.
# Compare qat2w_s$S with qat4_s$S from sweep_2b.sh. Seed from $1.
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
run qat2w_s$S --qbits 2 --d 192 --heads 6 --ffn 560
