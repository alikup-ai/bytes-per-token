#!/bin/sh
cd "$(dirname "$0")/../.."
P=.venv/Scripts/python
for m in combo_s0 qat4_s0 relu_s0 ple64_s0; do
  $P engine/export.py $m | tail -1
  runs=1; [ $m = combo_s0 ] && runs=3
  echo "== $m ($runs board runs)"
  $P engine/test.py $m --port COM6 --runs $runs 2>&1 | grep -v "^    ENG"
done
