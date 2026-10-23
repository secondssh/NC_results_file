"""relearning attack out-domain 데이터 (CLINC work 대상, seed 0). 형식은 out_banking77_1000.json 과 동일.
  out_massive_work_1000.json : forget 밖(가까운 공개 데이터) — MASSIVE (Amazon, Apache-2.0) en-US train 중
                               업무 비서 계열 8 intent 에서 1000 개 (intent 균등). 라벨 = MASSIVE intent (CLINC 라벨 체계와 다름)
  CLINC work 15 intent 중 schedule_meeting / meeting_schedule / next_holiday 와 의미가 가깝고 (calendar_*),
  급여·휴가·보험 계열은 MASSIVE 에 없음. CLINC 문장과 완전히 같은 문장은 제외.
원본: data/download/massive/en_train.json.gz (https://huggingface.co/datasets/mteb/amazon_massive_intent, train/en.json.gz)
python relearn_attack/make_data_massive.py
"""
import gzip
import json
import os
import random

from make_data import HERE, REPO, balanced, rec

INTENTS = ["calendar_set", "calendar_query", "calendar_remove", "datetime_query",
           "email_query", "email_sendemail", "email_querycontact", "email_addcontact"]


def main():
    rng = random.Random(0)
    rows = [json.loads(line) for line in gzip.open(f"{HERE}/data/download/massive/en_train.json.gz", "rt") if line.strip()]
    raw = json.load(open(f"{REPO}/data/clinc150/clinc150_uci/data_full.json"))
    clinc = {u.strip().lower() for split in raw.values() for u, _ in split}
    pool = [(r["text"].strip(), r["label"]) for r in rows
            if r["label"] in INTENTS and r["text"].strip().lower() not in clinc]
    d_out = [rec(u, i, "work") for u, i in balanced(pool, 1000, rng)]
    json.dump(d_out, open(f"{HERE}/data/out_massive_work_1000.json", "w"), indent=1, ensure_ascii=False)
    cnt = {i: sum(r["intent"] == i for r in d_out) for i in INTENTS}
    print(f"out: {len(d_out)} (pool {len(pool)}, CLINC 과 같은 문장 제외 "
          f"{sum(r['label'] in INTENTS for r in rows) - len(pool)}) | {cnt}")
    for r in d_out[:8]:
        print("  ", r["question"], "->", r["answer"])


if __name__ == "__main__":
    main()
