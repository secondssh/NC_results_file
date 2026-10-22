"""base 모델(unlearning 전) 원본/paraphrase 평가 → step 별 forget/retained acc 재구성 + RULE 과 비교.

base 는 step 마다 안 바뀌므로 run_base_para_eval.sh 가 전체 문항(마지막 step 의 SD + RD)을 1회 평가했다.
step t 의 SD/RD 파일에 든 문항만 골라 평균하면 그 step 의 base 점수가 된다.

문항별 정답 여부: 결과 json 의 eval_results 는 저장 때 평균으로 합쳐져 없어서 failure_text 로 복원한다
  (eval_util*: forget 은 '맞힌 것', retained 는 '틀린 것' 이 failure_text). 복원 평균 == choice_accuracy 를 assert.

python repro_check/base_para_steps.py   → repro_check/base_para_steps.{md,csv}
"""
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(os.path.dirname(ROOT), "data_new")
RES = os.path.join(ROOT, "run_results/eval_basemoe_error/llama3.1-8b-instruct")
BASE = os.path.join(RES, "para_eval/base")
TYPES = ["orig", "para", "para_ctx", "para_struct", "para_bt", "para_noise"]
CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]
N_STEPS = {"tofu": 20, "clinc": 9}


def _key(t):
    return json.dumps(t, ensure_ascii=False)


def per_sample_hits(split_name, res):
    """failure_text 로 문항별 hit(정답) 복원."""
    gen = res["generated_text"]
    fail = Counter(_key(x) for x in res.get("failure_text", []))
    hits = []
    for g in gen:
        k = _key(g)
        in_fail = fail[k] > 0
        if in_fail:
            fail[k] -= 1
        hits.append(in_fail if "forget" in split_name else not in_fail)
    acc = sum(hits) / len(hits)
    assert abs(acc - res["choice_accuracy"]) < 1e-9, (split_name, acc, res["choice_accuracy"])
    return hits


def step_files(ds, ty, t):
    suf = "" if ty == "orig" else f"_{ty}"
    if ds == "tofu":
        n = f"{t + 1:03d}"
        return (f"{DATA}/ptofu_SD{suf}/tofu_step{n}_SD.json", f"{DATA}/ptofu_RD{suf}/tofu_step{n}_RD.json")
    j = "_".join(CLINC_DOMAINS[:t + 1])
    return (f"{DATA}/clincqa_SD_10{suf}/clincqa_{j}_test_SD.json",
            f"{DATA}/clincqa_RD_10{suf}/clincqa_not_{j}_test_RD.json")


def qkey(rec):
    # paraphrase 는 서로 다른 원문이 같은 문장이 되기도 해서(CLINC para 153건) 원문+정답+도메인/저자로 식별.
    # 6개 데이터 모두 전체 문항에서 이 키는 중복 없음 (2026-10-20 확인).
    return (rec.get("orig_question", rec.get("question")), rec.get("answer"), rec.get("intent"),
            rec.get("domain"), rec.get("forget_author"))


def base_steps(ds, ty):
    f = os.path.join(BASE, f"{ds}_{ty}.json")
    if not os.path.isfile(f):
        return None
    d = json.load(open(f))
    pool_sd, pool_rd = step_files(ds, ty, N_STEPS[ds] - 1)
    hit_of = defaultdict(list)                # 문항 key -> hit (중복 문항이면 여러 개)
    for sp, pool in (("forget", pool_sd), ("retained", pool_rd)):
        recs = json.load(open(pool))
        hits = per_sample_hits(sp, d[sp]["results"])
        assert len(recs) == len(hits), (sp, len(recs), len(hits))
        for r, h in zip(recs, hits):
            hit_of[qkey(r)].append(h)
    assert all(len(v) == 1 for v in hit_of.values()), "문항 키 중복"
    out = {}
    for t in range(N_STEPS[ds]):
        row = {}
        for sp, path in zip(("forget", "retained"), step_files(ds, ty, t)):
            hs = [hit_of[qkey(r)][0] for r in json.load(open(path))]
            row[sp] = sum(hs) / len(hs)
        out[t] = row
    return out


def rule_steps(ds, ty):
    if ty == "orig":
        d = f"{RES}/repro_tofu_lunar/tofu_moe_fixed_layer16_0" if ds == "tofu" else \
            f"{RES}/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0"
    else:
        d = f"{RES}/para_eval/{ds}_{ty}/" + ("tofu_moe_fixed_layer16_0" if ds == "tofu" else "clincqa_moe_fixed_layer16_0")
    out = {}
    for f in glob.glob(f"{d}/lunar_*_t*_*/forget_*.json"):
        t = int(re.search(r"_t(\d+)_", os.path.basename(os.path.dirname(f))).group(1))
        r = json.load(open(f))
        out[t] = {sp: r[sp]["results"]["choice_accuracy"] for sp in ("forget", "retained")}
    return out


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def main():
    L = ["# base(unlearning 전) vs RULE — 원본/paraphrase, step 별 choice_accuracy (seed 0)\n",
         "base 는 전체 문항 1회 평가 후 step t 의 SD/RD 문항만 골라 평균. "
         "RULE − base: forget 은 음수일수록 잘 잊음, retained 는 0 에 가까울수록 보존.\n"]
    rows = []
    for ds in ["tofu", "clinc"]:
        L.append(f"\n## {ds.upper()}\n")
        L.append("### 요약 (RULE 이 끝난 step 평균)\n")
        L.append("| type | step | base forget | RULE forget | RULE−base | base retained | RULE retained | RULE−base |")
        L.append("|---|---|---|---|---|---|---|---|")
        per_type = {}
        for ty in TYPES:
            b, r = base_steps(ds, ty), rule_steps(ds, ty)
            per_type[ty] = (b, r)
            ts = sorted(t for t in r if b and t in b)
            for t in range(N_STEPS[ds]):
                rows.append(dict(ds=ds, type=ty, step=t,
                                 base_forget=b[t]["forget"] if b else None, rule_forget=r.get(t, {}).get("forget"),
                                 base_retained=b[t]["retained"] if b else None, rule_retained=r.get(t, {}).get("retained")))
            if not ts:
                L.append(f"| {ty} | {'base 없음' if not b else 'RULE 없음'} | | | | | | |")
                continue
            m = lambda src, sp: sum(src[t][sp] for t in ts) / len(ts)
            bf, rf, br, rr = m(b, "forget"), m(r, "forget"), m(b, "retained"), m(r, "retained")
            L.append(f"| {ty} | t{ts[0]}–t{ts[-1]} ({len(ts)}) | {bf:.3f} | {rf:.3f} | {rf - bf:+.3f} | "
                     f"{br:.3f} | {rr:.3f} | {rr - br:+.3f} |")
        for sp in ["forget", "retained"]:
            L.append(f"\n### {sp} — step 별 (base / RULE)\n")
            L.append("| step | " + " | ".join(TYPES) + " |")
            L.append("|---" * (len(TYPES) + 1) + "|")
            for t in range(N_STEPS[ds]):
                cells = []
                for ty in TYPES:
                    b, r = per_type[ty]
                    cells.append(f"{f3(b[t][sp] if b else None)} / {f3(r.get(t, {}).get(sp))}")
                L.append(f"| t{t} | " + " | ".join(cells) + " |")
    out = os.path.join(ROOT, "repro_check/base_para_steps")
    open(out + ".md", "w").write("\n".join(L) + "\n")
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out + ".md")


if __name__ == "__main__":
    main()
