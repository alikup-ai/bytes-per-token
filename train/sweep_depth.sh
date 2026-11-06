#!/bin/sh
# Depth vs width at equal weight bytes read per token (~901 KB at 4-bit,
# same budget as sweep_h2.sh). Head dim 32 everywhere; FFN sized to hit the
# budget. dense_s$S from sweep_h2.sh is the 6-layer arm. Seed from $1.
# Cost model (scenario 5, no overlap): L4 19.1, L6 20.6, L12 23.7 ms/token.
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
run depth4_s$S  --layers 4  --d 160 --heads 5 --ffn 384
run depth12_s$S --layers 12 --d 96  --heads 3 --ffn 272
