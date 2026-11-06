#!/bin/sh
# Put the demo on a board: demo firmware + a model pack. Usage:
#   engine/demo/flash_demo.sh COM6 [esp32ai_ts|combo_s0] [build dir: builddemo]
# Needs the ESP-IDF build of the demo (see README.md) and the pack in engine/out/.
set -e
cd "$(dirname "$0")/../.."
PORT=${1:-COM6}; MODEL=${2:-esp32ai_ts}; BUILD=${3:-builddemo}
P=.venv/Scripts/python
(cd engine/esp32/$BUILD && ../../../$P -m esptool --chip esp32s3 -p $PORT -b 921600 write_flash @flash_args)
$P -m esptool --chip esp32s3 -p $PORT -b 921600 write_flash 0x90000 engine/out/$MODEL.pack
rm -f engine/out/.flashed_$PORT
echo "done: open engine/demo/index.html in Chrome or Edge and press Connect"
