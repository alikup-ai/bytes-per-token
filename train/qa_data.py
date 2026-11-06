"""Question-answer data built from TinyStories by rules (no other model involved).

Each example is a story followed by one question about it and a one-sentence
answer taken from the story:

    <story>
    Question: Where did Tom go?
    Answer: Tom went to the park.<|endoftext|>

A question is asked only when the story answers it exactly once (for example,
exactly one "Tom went to ..." sentence), so every answer is unambiguous and
grounded in the text. Question types: name of a character, who a character is,
where they went, what they found / saw / had / wanted / liked / loved / got /
made / ate, how they felt, what they said.

    .venv/Scripts/python train/qa_data.py

Writes data/qa_{train,val}.bin (uint16 tokens, rows of SEQ tokens, padded with
end-of-text), data/qa_{train,val}.mask.bin (uint8, 1 on answer tokens: the
loss is taken there only) and data/qa_val.jsonl (prompt and gold answer, for
exact-match evaluation of generated answers).
"""
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

DATA = Path(__file__).resolve().parent.parent / "data"
EOT = "<|endoftext|>"
SEQ = 256
N_TRAIN_STORIES = 250_000

NOUNS = ("girl|boy|dog|cat|bird|bunny|rabbit|bear|mouse|fish|frog|puppy|kitten|duck|lion|monkey|"
         "man|woman|lady|car|truck|squirrel|owl|fox|elephant|horse|cow|pig|sheep|butterfly|bee|ant|robot|dragon|turtle")
NAMED = re.compile(r"\b[Aa]n? ((?:[a-z]+ ){0,3}?)(" + NOUNS + r") named ([A-Z][a-z]+)\b")
VERBS = {"went": "go", "ran": "run", "walked": "walk", "flew": "fly", "swam": "swim"}
OBJ_VERBS = {"found": "find", "saw": "see", "had": "have", "wanted": "want", "liked": "like", "loved": "love",
             "got": "get", "made": "make", "ate": "eat", "lost": "lose", "built": "build", "drew": "draw"}
FEEL = "happy|sad|scared|angry|excited|surprised|tired|proud|hungry|upset|worried|lonely|curious|shy|bored|mad|afraid|glad|sorry"


def stories(path, limit=None):
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = [s.strip() for s in text.split(EOT)]
    parts = [p for p in parts[:-1] if p]
    return parts[:limit] if limit else parts


def sentences(story):
    return re.split(r"(?<=[.!?])\s+|\n+", story)


def short(phrase, max_words=8):
    phrase = phrase.strip()
    return phrase if 0 < len(phrase.split()) <= max_words else None


def once(matches):
    """The single distinct match, or None when there are zero or several."""
    distinct = list(dict.fromkeys(matches))
    return distinct[0] if len(distinct) == 1 else None


def make_qa(story, names_known):
    out = []
    named = NAMED.findall(story)
    chars = {}
    for adj, noun, name in named:
        chars.setdefault(name, (adj.strip(), noun))
    # who is X / what was the name of the Y
    for name, (adj, noun) in chars.items():
        if sum(1 for _, n, _ in named if n == noun) == 1:
            out.append((f"What was the name of the {noun}?", f"The {noun} was named {name}."))
        desc = f"{adj} {noun}".strip()
        art = "an" if desc[0] in "aeiou" else "a"
        out.append((f"Who is {name}?", f"{name} is {art} {desc}."))
    names = set(chars) | (set(re.findall(r"\b[A-Z][a-z]+\b", story)) & names_known)
    sents = sentences(story)
    for name in names:
        # where did X go
        m = once([s for s in sents for s in re.findall(r"\b" + name + r" (went|ran|walked|flew|swam) (to [^.,!?;\"]+)", s)])
        if m and short(m[1]):
            out.append((f"Where did {name} {VERBS[m[0]]}?", f"{name} {m[0]} {m[1].strip()}."))
        # what did X find / see / ...
        for past, base in OBJ_VERBS.items():
            m = once([x for s in sents for x in re.findall(r"\b" + name + r" " + past + r" ((?:a|an|the|his|her|some) [^.,!?;\"]+)", s)])
            if m and short(m, 6):
                out.append((f"What did {name} {base}?", f"{name} {past} {m.strip()}."))
        # how did X feel
        m = once(re.findall(r"\b" + name + r" (?:was|felt) (?:very |so |really )?(" + FEEL + r")\b", story))
        if m:
            out.append((f"How did {name} feel?", f"{name} felt {m}."))
        # what did X say
        m = once(re.findall(r"\b" + name + r" said,? \"([^\"]+)\"", story) +
                 re.findall(r"\"([^\"]+)\",? " + name + r" said", story))
        if m and short(m, 12):
            q = m.strip().rstrip(",")
            if q[-1] not in ".!?":
                q += "."
            out.append((f"What did {name} say?", f"{name} said, \"{q}\""))
    return out


def build(tok, items, rng, per_story, jsonl=None):
    eot = tok.token_to_id(EOT)
    rows, masks, kinds = [], [], Counter()
    for story, qas in items:
        if not qas:
            continue
        rng.shuffle(qas)
        for q, a in qas[:per_story]:
            prompt = f"{story}\nQuestion: {q}\nAnswer:"
            p = tok.encode(prompt).ids
            ans = tok.encode(" " + a).ids + [eot]
            if len(p) + len(ans) > SEQ + 1:
                continue
            ids = p + ans
            row = ids + [eot] * (SEQ + 1 - len(ids))
            mask = [0] * len(p) + [1] * len(ans) + [0] * (SEQ + 1 - len(ids))
            rows.append(row)
            masks.append(mask)
            kinds[q.split()[0] + " " + q.split()[1]] += 1
            if jsonl is not None:
                jsonl.write(json.dumps(dict(prompt=prompt, answer=a, question=q)) + "\n")
    return np.array(rows, dtype=np.uint16), np.array(masks, dtype=np.uint8), kinds


def main():
    tok = Tokenizer.from_file(str(DATA / "tok4096.json"))
    rng = random.Random(0)
    train = stories(DATA / "train_part.txt", N_TRAIN_STORIES)
    val = stories(DATA / "valid.txt")
    names = Counter(n for s in train[:100_000] for _, _, n in NAMED.findall(s))
    names_known = {n for n, c in names.most_common(150) if c >= 20}
    print(f"{len(names_known)} common names, e.g. {sorted(names_known)[:12]}")

    tr_items = [(s, make_qa(s, names_known)) for s in train]
    va_items = [(s, make_qa(s, names_known)) for s in val]
    X, Mk, k_tr = build(tok, tr_items, rng, per_story=2)
    with open(DATA / "qa_val.jsonl", "w", encoding="utf-8") as f:
        Xv, Mv, k_va = build(tok, va_items, rng, per_story=1, jsonl=f)
    X.tofile(DATA / "qa_train.bin"); Mk.tofile(DATA / "qa_train.mask.bin")
    Xv.tofile(DATA / "qa_val.bin"); Mv.tofile(DATA / "qa_val.mask.bin")
    print(f"train {len(X):,} examples, val {len(Xv):,}; rows of {SEQ + 1} tokens")
    print("question types (train):", dict(k_tr.most_common()))
    for s, qas in va_items[:3]:
        print("\n" + s[:300].replace("\n", " ") + " ...")
        for q, a in qas[:4]:
            print(f"  Q: {q}  A: {a}")


if __name__ == "__main__":
    main()
