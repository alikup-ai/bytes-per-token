#!/bin/sh
# PSRAM 80 vs 120 MHz on the board (docs/phase1-bench.md). Usage: engine/psram120.sh COM6
# 1) raw bandwidth: bench firmware at 120 MHz -> bench/roofline_120.json
# 2) engine speed: combo and esp32-ai, 80 MHz build then 120 MHz build
set -e
cd "$(dirname "$0")/.."
PORT=${1:-COM6}
P=.venv/Scripts/python
echo "== bench, PSRAM 120 MHz"
(cd bench/esp32s3_bench/build120 && ../../../$P -m esptool --chip esp32s3 -p $PORT -b 921600 write_flash @flash_args > /dev/null)
rm -f engine/out/.flashed_$PORT                       # the bench image overwrites part of the model partition
$P bench/capture.py --port $PORT | grep -E "BENCH|wrote"
cp bench/roofline.json bench/roofline_120.json
for build in build build120; do
  for m in combo_s0 esp32ai_ts; do
    echo "== engine $build $m"
    $P engine/test.py $m --port $PORT --flash-app --build $build --tag _$build --runs 1 | grep -E "PASS\] run|FAIL|speed pie|batch \(|checks"
  done
done
