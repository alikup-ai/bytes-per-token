#!/bin/sh
# Variant A (2026-09-29): combo with a PLE table twice as wide (P 64 -> 128), FFN 320 -> 256
# so the bytes read per token stay equal to combo (1.39M params read with the two-stage head).
# Question: does a bigger flash table buy quality at the same speed? Compare with combo_s0/s1.
set -e
cd "$(dirname "$0")"
P=../.venv/Scripts/python
T=20e6
run() {
    if [ -f "../runs/$1/final.json" ]; then echo "skip $1"; return; fi
    n=$1; shift
    $P train.py --name "$n" --tokens $T "$@"
}
for S in 0 1; do
    run comboP128_s$S --seed $S --ffn 256 --ple 128 --act relu --qbits 4
done
