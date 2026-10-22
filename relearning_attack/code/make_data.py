"""relearning attack 데이터 (CLINC banking 대상, seed 0). 형식은 data_new/clincqa_SD_only_200/*_train_SD.json 과 동일.

  in_banking_100.json   : forget 안 — CLINC banking train 중 unlearning(clincqa_SD_only_200) 에 안 쓴 문장 100개
                          (15 intent 균등, test 와 겹치지 않음). 라벨 = CLINC intent
  out_banking77_1000.json : forget 밖(가까운 공개 데이터) — BANKING77 train 1000개 (77 intent 균등).
                          라벨 = BANKING77 category (CLINC 라벨 체계와 다름)
python relearn_attack/make_data.py
"""
import csv
import json
import os
import random
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
INST = "Classify the user's intent from the following utterance. Return only the intent label.\n\nUtterance: "


def rec(u, intent, domain):
    ti = INST + u
    return {"question": u, "intent": intent, "domain": domain, "instruction": ti,
            "output": f"Intent: {intent}", "input": "", "answer": intent,
            "text_full": f"{ti}Intent: {intent}", "text_input": ti}


def balanced(items, n, rng):
    by = defaultdict(list)
    for u, i in items:
        by[i].append(u)
    for v in by.values():
        rng.shuffle(v)
    out, k = [], 0
    while len(out) < n:
        for i in sorted(by):
            if k < len(by[i]) and len(out) < n:
                out.append((by[i][k], i))
        k += 1
    rng.shuffle(out)
    return out


def main():
    rng = random.Random(0)
    raw = json.load(open(f"{REPO}/data/clinc150/clinc150_uci/data_full.json"))
    bank = set(json.load(open(f"{REPO}/data/clinc150/CLINC_domain.json"))["banking"])
    used = {r["question"] for r in json.load(open(f"{REPO}/data_new/clincqa_SD_only_200/clincqa_banking_train_SD.json"))}
    test = {u for u, i in raw["test"] if i in bank}
    pool = [(u, i) for u, i in raw["train"] if i in bank and u not in used and u not in test]
    d_in = [rec(u, i, "banking") for u, i in balanced(pool, 100, rng)]

    with open(f"{HERE}/data/banking77_train.csv") as f:
        b77 = [(r["text"].strip(), r["category"]) for r in csv.DictReader(f)]
    d_out = [rec(u, i, "banking") for u, i in balanced(b77, 1000, rng)]

    os.makedirs(f"{HERE}/data", exist_ok=True)
    json.dump(d_in, open(f"{HERE}/data/in_banking_100.json", "w"), indent=1, ensure_ascii=False)
    json.dump(d_out, open(f"{HERE}/data/out_banking77_1000.json", "w"), indent=1, ensure_ascii=False)
    print(f"in: {len(d_in)} ({len({r['intent'] for r in d_in})} intents, pool {len(pool)}) | "
          f"out: {len(d_out)} ({len({r['intent'] for r in d_out})} intents)")
    print("in ex :", d_in[0]["question"], "->", d_in[0]["answer"])
    print("out ex:", d_out[0]["question"], "->", d_out[0]["answer"])


if __name__ == "__main__":
    main()
