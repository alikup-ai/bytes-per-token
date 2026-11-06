"""Ask the question-answering model about a story, on the PC (no board needed).

    .venv/Scripts/python train/ask.py                      # built-in story, type your questions
    .venv/Scripts/python train/ask.py --story my_story.txt # your own story (English, up to ~150 words)
    .venv/Scripts/python train/ask.py -q "Who is Lily?"    # one question and exit

Same model as the board (runs/combo_qa_s0), same prompt format:
"<story>\\nQuestion: <q>\\nAnswer:". Greedy answer, one line.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from finetune_qa import answer
from model import LM, Config

ROOT = Path(__file__).resolve().parent.parent
STORY = ("Once upon a time, there was a little girl named Lily. She lived in a small house near the forest. "
         "One day, Lily went to the park with her mom. She found a shiny red ball under a big tree. "
         "Lily was very happy. She said, \"Mom, look at my new ball!\" They played with the ball until the sun went down.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="combo_qa_s0")
    ap.add_argument("--story", help="text file with the story")
    ap.add_argument("-q", "--question", action="append")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    meta = json.loads((ROOT / "runs" / a.run / "final.json").read_text())
    model = LM(Config(**meta["cfg"]))
    model.load_state_dict(torch.load(ROOT / "runs" / a.run / "model.pt", weights_only=True))
    model.eval()
    tok = Tokenizer.from_file(str(ROOT / "data" / "tok4096.json"))
    story = Path(a.story).read_text(encoding="utf-8").strip() if a.story else STORY
    n = len(tok.encode(story).ids)
    print(f"Story ({n} tokens):\n{story}\n")
    if n > 210:
        print("The story is too long for the model's 256-token window; shorten it to ~150 words.")
        return

    def reply(q):
        return answer(model, tok, story + "\nQuestion: " + q + "\nAnswer:")

    def ask(q):
        print(f"Q: {q}\nA: {reply(q)}\n")

    if a.question:
        for q in a.question:
            ask(q)
        return
    print("Type a question in English (empty line to quit). Examples: Who is Lily? / Where did Lily go? /"
          " What did Lily find? / How did Lily feel? / What did Lily say?\n")
    while True:
        try:
            q = input("Q: ").strip()
        except EOFError:
            break
        if not q:
            break
        print(f"A: {reply(q)}\n")


if __name__ == "__main__":
    main()
