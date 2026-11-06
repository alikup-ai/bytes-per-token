#!/bin/sh
# Runs added after the 2026-09-29 review:
#   combo_s1, relu_qat4_s1 : second seed for the two single-seed arms the report relies on
#   qat2w_lr15_s0          : 2 bits at equal bytes with a lower LR (lr 3e-3 was never tuned for d=192;
#                            qat2w's seed spread was 0.10)
# Arms with a final.json are skipped; killed arms resume from ckpt.pt.
set -e
cd "$(dirname "$0")"
P=../.venv/Scripts/python
T=20e6
run() {
    if [ -f "../runs/$1/final.json" ]; then echo "skip $1"; return; fi
    n=$1; shift
    $P train.py --name "$n" --tokens $T "$@"
}
run combo_s1      --seed 1 --ffn 320 --ple 64 --act relu --qbits 4
run relu_qat4_s1  --seed 1 --act relu --qbits 4
run qat2w_lr15_s0 --seed 0 --qbits 2 --d 192 --heads 6 --ffn 560 --lr 1.5e-3
