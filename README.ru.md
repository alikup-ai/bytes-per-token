# mcu-lm-research

Маленькая языковая модель на ESP32-S3 N16R8: сколько байт весов нужно прочитать на один токен и как сделать их меньше.
Результаты по этапам: `docs/phase*.md`. Сводный отчёт: `docs/report.html`.

## Структура
| папка | что там |
|---|---|
| `costmodel/` | модель затрат (мс/токен по операциям), калибровка `validate.py`, сценарии `whatif.py`, `ideas.py` |
| `bench/` | прошивка-бенчмарк платы (память, PIE, DMA, шина PSRAM) и логи замеров |
| `train/` | PyTorch: модели, обучение, sweep-скрипты, данные для вопросов по сказке |
| `engine/` | движок на C (общий для ПК и платы), прошивки ESP-IDF, экспорт моделей, тесты, демо |
| `decode/` | симуляция спекулятивного декодирования |
| `docs/` | журнал экспериментов, отчёт |

## Данные (не в git: `data/`, `runs/`, `refs/` в `.gitignore`)
- `data/train_part.txt` — первые 300 МиБ (314 572 800 байт) файла `TinyStoriesV2-GPT4-train.txt`, `data/valid.txt` — `TinyStoriesV2-GPT4-valid.txt` целиком (оба из `roneneldan/TinyStories` на Hugging Face; сверено побайтно и по SHA-256 06.11.2026).
- `data/tok4096.json`, `train.bin`, `val.bin` — `train/data.py` (BPE на 4096 токенов по первым 200k историям).
- `data/qa_*` — `train/qa_data.py` (вопросы по сказке, правила, без другой модели).
- `refs/esp32-ai`, `refs/esp32-tinyLLM` — клоны сравниваемых проектов (коммит esp32-ai 52ebcbb, см. `docs/phase3-esp32ai.md`).
- Модель esp32-ai скачивается `engine/export_esp32ai.py` с проверкой SHA-256.

## Воспроизведение
```sh
python -m venv .venv && .venv/Scripts/pip install torch tokenizers numpy scipy pyserial esptool
.venv/Scripts/python train/data.py                  # токенизатор и train/val.bin
sh train/sweep_h2.sh 0; sh train/sweep_h2.sh 1      # и другие sweep_*.sh; ~30-50 мин на модель на CPU
python costmodel/validate.py && python costmodel/ideas.py
.venv/Scripts/python engine/test.py combo_s0                  # ПК: движок против PyTorch
.venv/Scripts/python engine/test.py combo_s0 --port COM6 --flash-app   # плата (сборка: engine/esp32, ESP-IDF 5.5.5)
```

## Известные ограничения
- Модели маленькие (2–29 млн параметров), обучены на детских рассказах, 20M токенов на модель, один LR для всех форм.
- Скорость зависит от длины контекста: у combo 113 ток/с на позициях 13–31 и 72 ток/с на 224–255 (`engine/test.py`, строка poscurve).
- PSRAM 120 МГц — экспериментальный режим ESP-IDF.
- Второе ядро занято активным ожиданием, сторожевой таймер задач выключен (`engine/esp32/sdkconfig.defaults`):
  для продукта это нужно заменить; энергия на токен не измерялась.
