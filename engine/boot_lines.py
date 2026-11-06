"""Reset the board and print the ENG lines it writes before the tests start.
    .venv/Scripts/python engine/boot_lines.py COM6 [stop_after_test_name]"""
import sys
import time

import serial

port = sys.argv[1]
stop = sys.argv[2] if len(sys.argv) > 2 else "micro"
s = serial.Serial(port, 115200, timeout=1)
s.dtr, s.rts = False, True
time.sleep(0.1)
s.rts = False
t0 = time.time()
while time.time() - t0 < 30:
    ln = s.readline().decode("utf-8", "replace").rstrip()
    if "ENG {" in ln or "Guru" in ln:
        print(ln)
    if f'"test":"{stop}"' in ln or "Guru" in ln:
        break
