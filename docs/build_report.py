"""Build docs/report.html from docs/report.src.html (edit the source, not the output).

    python docs/build_report.py

The source holds the page with `const RACE = __RACE__;`; this fills in the
token race (the same 200 tokens both engines generated on esp32-ai's benchmark,
docs/phase3-esp32ai.md) from docs/report_race.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    src = (HERE / "report.src.html").read_text(encoding="utf-8")
    race = json.loads((HERE / "report_race.json").read_text(encoding="utf-8"))
    assert src.count("__RACE__") == 1
    html = src.replace("__RACE__", json.dumps(race, ensure_ascii=False))
    (HERE / "report.html").write_text(html, encoding="utf-8", newline="\r\n")
    print("wrote docs/report.html")


if __name__ == "__main__":
    main()
