# Questions about a story (2026-09-28)

The TinyStories models only continue text. `combo_qa_s0` is `combo_s0` fine-tuned to answer simple questions
about a story given in the prompt:

    <story>
    Question: Where did Tom go?
    Answer: Tom went to the park.<|endoftext|>

## Data (`train/qa_data.py`)
Made by rules from TinyStories, no other model: 280k training examples (250k stories, up to 2 questions each),
16k held-out ones from the validation stories. A question is asked only when the story answers it exactly
once, and the answer sentence is built from the story's own words. Types: name of a character, who a
character is, where they went, what they found / saw / had / wanted / liked / loved / got / made / ate / lost,
how they felt, what they said. Rows are 257 tokens; the loss is taken on the answer tokens only.

## Training (`train/finetune_qa.py --base combo_s0 --steps 2000`)
Batches of 32: 24 question rows + 8 plain story windows (so story writing is kept). AdamW, lr 1e-3 cosine,
QAT 4-bit as the base. 2000 steps, 50 min on the laptop CPU.

| step | exact match (held-out) | story val loss |
|---|---|---|
| 0 (base model) | 0% (it continues the story) | 1.997 |
| 500 | 81% | 2.310 |
| 1000 | 85% | 2.260 |
| 1500 | 88% | 2.194 |
| 2000 (600 questions) | **89%** | 2.174 |

Per type at step 2000: name 98%, feeling 95%, who 91%, where 86%, what did X <verb> 79%, what did X say 21%.
Quoting a long sentence word for word is the hard case. Questions of other kinds ("What color was the ball?")
get an answer in the form of a trained kind. Story writing costs +0.18 val loss.

## On the board
- `engine/test.py combo_qa_s0 --flash-app`: 11/11 checks, board == PC bit-exact, 107 tok/s (2 cores, 80 MHz).
- `engine/qa_test.py` asks 100 held-out questions through the demo firmware (`engine/out/qa_board*.log`):

| head | exact match | board answer == PyTorch answer | answer speed |
|---|---|---|---|
| two-stage, 15% of rows | 84% | 94/100 | 13.0 ms/token |
| **full** | **87%** | **99/100** | 15.5 ms/token |

The two-stage head drops rare pieces of names ("Binky" -> "B" + "y", "Roy" -> "Ro"): an answer copies names
from the story, and those name tokens sit in clusters the screening does not read. The demo therefore sends
`H 1` (full head) before a question and `H 0.15` before story writing; a short answer costs ~20 ms more.
Reading a story of ~200 tokens takes ~1.1 s (batched), then the answer streams at ~64-77 tok/s (long context).

## Limits of these numbers (review 2026-09-29)
- The answers are made by the same rules as the training data and copy a sentence of the story. Exact match
  measures how well the model extracts and copies in these ~12 templates, not open question answering.
- Only stories that fit 257 tokens with the question and answer are in the data, so all numbers are for short stories.
- The training-time table uses 200 random held-out questions at steps 500-1500 and 600 at step 2000 (different
  samples): ±3-4 points of sampling noise per row.
- The board test asks 100 questions from those whose prompt fits 240 tokens (another subset than the 600).
  84% vs 87% (two-stage vs full head) is 3 questions, inside the noise of 100 questions (±7 points). The
  trustworthy board number is agreement with PyTorch on the same questions: 94/100 and 99/100.
- One seed; story writing gets worse by +0.18 val loss (2.00 -> 2.17), a real cost of the fine-tune.

## Demo
`engine/demo/index.html`, mode "Вопрос по сказке", with the demo firmware and `combo_qa_s0`
(`engine/demo/flash_demo.sh COM6 combo_qa_s0`). On the PC without a board: `.venv/Scripts/python train/ask.py`.
