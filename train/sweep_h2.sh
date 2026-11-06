#!/bin/sh
# H2, first pass: equal weight bytes read per token (~901 KB at 4-bit),
# different places to store extra parameters. Seed from $1 (default 0).
# Sequential on purpose: parallel runs would share 8 CPU threads and
# make the elapsed times meaningless. Arms with a final.json are skipped,
# so a killed sweep can simply be restarted.
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
run dense_s$S
run dense_f320_s$S --ffn 320
run ple64_s$S      --ffn 320 --ple 64
run moe4_s$S       --experts 4
