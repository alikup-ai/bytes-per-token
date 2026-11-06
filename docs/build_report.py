"""Build docs/report.html (English) and docs/report.ru.html (Russian) from their .src.html sources.

    python docs/build_report.py

Each source holds the page with `const RACE = __RACE__;`; this fills in the
token race (the same 200 tokens both engines generated on esp32-ai's benchmark,
docs/phase3-esp32ai.md) from docs/report_race.json.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent


def main():
    race = json.loads((HERE / "report_race.json").read_text(encoding="utf-8"))
    for src_name, out_name in (("report.src.html", "report.html"), ("report.ru.src.html", "report.ru.html")):
        src = (HERE / src_name).read_text(encoding="utf-8")
        assert src.count("__RACE__") == 1
        html = src.replace("__RACE__", json.dumps(race, ensure_ascii=False))
        (HERE / out_name).write_text(html, encoding="utf-8", newline="\r\n")
        print("wrote docs/" + out_name)


if __name__ == "__main__":
    main()
