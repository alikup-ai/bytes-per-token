"""README GIF: the same 200 tokens typed by esp32-ai's firmware and by our engine, at the measured speeds.

    python docs/media/race_gif.py docs/media/race.gif

Text: their firmware output (bench/esp32ai_fw_run1.log), which our engine reproduces character for character.
Speeds: their firmware 94.10 ms/token, our engine 11.52 ms/token (two-stage head, PSRAM 80 MHz),
both measured on the same ESP32-S3-N16R8, positions 4-203 (docs/phase3-esp32ai.md).
"""
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
W, H = 1920, 1080          # drawn at 2x, saved at 960x540
FPS = 15
RUN_S = 10.0               # real-time part
HOLD_S = 3.0               # final card
MS = {"theirs": 94.10, "ours": 11.52}
N_TOK = 200

HAND = r"C:\Windows\Fonts\segoepr.ttf"
HAND_B = r"C:\Windows\Fonts\segoeprb.ttf"
MONO = r"C:\Windows\Fonts\consola.ttf"
BOARD = (38, 42, 40)
PANEL = (28, 31, 30)
WHITE, CYAN, CORAL, GREY, LEMON = (240, 234, 218), (110, 215, 225), (240, 140, 115), (150, 150, 140), (240, 220, 120)


def story():
    log = (ROOT / "bench/esp32ai_fw_run1.log").read_text(encoding="utf-8", errors="replace")
    text = log.split(">>> ", 1)[1].split("\n--- ", 1)[0].strip()
    return text


def pieces(text):
    """Split into word-like pieces and map 200 tokens onto them (token i shows the first k pieces)."""
    parts = re.findall(r"\s*[A-Za-z']+|\s*[^\sA-Za-z']|\s+", text)
    ends, pos = [], 0
    for p in parts:
        pos += len(p)
        ends.append(pos)
    return [ends[min(len(ends) - 1, round((i + 1) * len(ends) / N_TOK) - 1)] for i in range(N_TOK)]


def wrap(text, font, width):
    lines = []
    for para in text.split("\n"):
        cur = ""
        for word in para.split(" "):
            t = (cur + " " + word) if cur else word
            if font.getlength(t) <= width:
                cur = t
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
    return lines


def frame(text, ends, T, final):
    img = Image.new("RGB", (W, H), BOARD)
    d = ImageDraw.Draw(img)
    hb = ImageFont.truetype(HAND_B, 52)
    h = ImageFont.truetype(HAND, 34)
    hs = ImageFont.truetype(HAND, 30)
    mono = ImageFont.truetype(MONO, 27)
    d.text((W // 2, 30), "Same ESP32-S3, same weights, same text, real time", font=hb, fill=WHITE, anchor="ma")

    panels = [("theirs", "esp32-ai firmware", CORAL, 40), ("ours", "bytes-per-token engine", CYAN, 980)]
    for key, label, color, x in panels:
        x1 = x + 900
        d.rounded_rectangle((x, 130, x1, 1000), 18, fill=PANEL, outline=color, width=3)
        n = min(N_TOK, int(T * 1000 / MS[key]))
        shown = text[:ends[n - 1]] if n else ""
        d.text((x + 24, 146), label, font=h, fill=color)
        rate = 1000 / MS[key]
        d.text((x1 - 24, 152), f"{rate:.1f} tok/s", font=h, fill=color, anchor="ra")
        y = 215
        for line in wrap(shown, mono, 852)[-25:]:
            d.text((x + 24, y), line, font=mono, fill=WHITE)
            y += 31
        if n < N_TOK and int(T * 2) % 2 == 0:
            d.rectangle((x + 24 + mono.getlength(wrap(shown, mono, 852)[-1]), y - 30, x + 38 + mono.getlength(wrap(shown, mono, 852)[-1]), y - 4), fill=color)
        done = f"{n}/{N_TOK} tokens" + (f"  -  done in {N_TOK * MS[key] / 1000:.1f} s" if n == N_TOK else "")
        d.text((x + 24, 955), done, font=hs, fill=GREY if n < N_TOK else color)

    if final:
        d.rounded_rectangle((560, 430, 1360, 650), 24, fill=BOARD, outline=LEMON, width=4)
        d.text((W // 2, 455), "8.2x faster", font=ImageFont.truetype(HAND_B, 80), fill=LEMON, anchor="ma")
        d.text((W // 2, 580), "94.1 -> 11.5 ms per token", font=h, fill=WHITE, anchor="ma")
    return img.resize((W // 2, H // 2), Image.LANCZOS)


def main(out):
    text = story()
    ends = pieces(text)
    frames = []
    for i in range(int(RUN_S * FPS)):
        frames.append(frame(text, ends, i / FPS, False))
    last = frame(text, ends, RUN_S, True)
    frames += [last] * int(HOLD_S * FPS)
    pal = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    print(out, len(frames), "frames", Path(out).stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main(sys.argv[1])
