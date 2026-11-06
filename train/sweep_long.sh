#!/bin/sh
# varC trained longer (2026-09-30): 70M tokens instead of 20M (~0.9 epoch of train_v8192.bin).
# varG showed the PLE table saturates at 20M tokens; is data the next lever? Compare with varC_s0 (0.7049 bpb).
set -e
cd "$(dirname "$0")"
P=../.venv/Scripts/python
if [ -f ../runs/varC_long_s0/final.json ]; then echo "skip"; exit 0; fi
$P train.py --name varC_long_s0 --seed 0 --tokens 70e6 --vocab 8192 --ffn 256 --ple 128 --act relu --qbits 4
