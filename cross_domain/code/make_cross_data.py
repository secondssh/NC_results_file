#!/usr/bin/env python3
"""Cross domain / cross author 입력 평가 세트 생성 (F+R, R+F). 학습 없음, GPU 불필요.

시점 t 의 원래 평가 파일(SD/RD)에서 forget(F) 문항 1개와 retain(R) 문항 1개를 짝지어 한 입력으로 합친다.
  CLINC t (4, 8) : F = clincqa_SD_10/clincqa_<도메인 0..t>_test_SD.json, R = clincqa_RD_10/clincqa_not_<...>_test_RD.json
             항상 450쌍: F 는 지운 도메인마다 450/(t+1) 개 층화, R 은 남은 도메인마다 같은 수 층화 (t8: R = meta 450 전부)
             → t4 = 5 × 90 / 5 × 90,  t8 = 9 × 50 / meta 450
  TOFU t (10, 19) : F = ptofu_SD/tofu_step{t+1}_SD.json 전부 (AG01..AG(t+1), 저자당 40), R = ptofu_RD/tofu_step{t+1}_RD.json
             F 각각에 R 1개 (중복 없이 추출) → t10 440쌍, t19 800쌍
두 순서(FR, RF)는 같은 쌍을 쓰고 순서만 바꾼다.

평가 파일의 주 필드(answer/intent/domain/perturbed_answer ...)는 F 쪽 값이다 → 기존 eval 을 "forget" split 으로
돌리면 저장되는 choice_accuracy = F 누출, router target = F 의 expert. R 쪽 정답은 각 문항의 "cross" 필드에 두고
score_cross.py 가 같은 생성문으로 따로 채점한다.

연결 템플릿 (고정)
  CLINC : "{a}, and also {b}"   (프롬프트 "Classify ... Return only the intent label.\n\nUtterance: " 는 그대로)
  TOFU  : "{a} Also, {b}"

출력: multi_attack/data/{clinc_t4,clinc_t8,tofu_t10,tofu_t19}_{FR,RF}.json, manifest.json
사용: python multi_attack/make_cross_data.py [--seed 1234] [--overwrite]
"""
import argparse
import hashlib
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "..", "data_new"))
OUT = os.path.join(HERE, "data")

CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]
STEPS = {"clinc": [4, 8], "tofu": [10, 19]}
TEMPLATE = {"clinc": "{a}, and also {b}", "tofu": "{a} Also, {b}"}
CLINC_N_PAIRS = 450


def src_paths(ds, t):
    if ds == "clinc":
        j = "_".join(CLINC_DOMAINS[:t + 1])
        return dict(F=f"{DATA}/clincqa_SD_10/clincqa_{j}_test_SD.json",
                    R=f"{DATA}/clincqa_RD_10/clincqa_not_{j}_test_RD.json")
    return dict(F=f"{DATA}/ptofu_SD/tofu_step{t + 1:03d}_SD.json",
                R=f"{DATA}/ptofu_RD/tofu_step{t + 1:03d}_RD.json")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def part(d, idx, side):
    """원 문항에서 채점/분석에 필요한 필드만."""
    keep = ("question", "answer", "intent", "domain", "paraphrased_answer", "perturbed_answer",
            "author_idx", "forget_author")
    return {"side": side, "src_idx": idx, **{k: d[k] for k in keep if k in d}}


def stratified(data, n, rng):
    doms = sorted({d["domain"] for d in data}, key=lambda x: (CLINC_DOMAINS + [x]).index(x))
    by_dom = {dom: [i for i, d in enumerate(data) if d["domain"] == dom] for dom in doms}
    assert n % len(doms) == 0, (n, doms)
    return [i for dom in doms for i in rng.sample(by_dom[dom], n // len(doms))]


def pair_clinc(F, R, rng):
    f_idx = stratified(F, CLINC_N_PAIRS, rng)
    rng.shuffle(f_idx)
    r_idx = list(range(len(R))) if len(R) == CLINC_N_PAIRS else stratified(R, CLINC_N_PAIRS, rng)
    return list(zip(f_idx, r_idx))


def pair_tofu(F, R, rng):
    r_idx = rng.sample(range(len(R)), len(F))
    return list(zip(range(len(F)), r_idx))


def build_clinc(f, r, order, pid):
    prefix = f["text_input"][: len(f["text_input"]) - len(f["question"])]
    assert f["text_input"].endswith(f["question"]) and r["text_input"].startswith(prefix)
    a, b = (f, r) if order == "FR" else (r, f)
    q = TEMPLATE["clinc"].format(a=a["question"], b=b["question"])
    item = dict(f)  # F 값 유지: intent / domain / answer / output
    item.update(question=q, instruction=prefix + q, text_input=prefix + q, text_full=prefix + q + f["output"])
    return item, q


def build_tofu(f, r, order, pid):
    a, b = (f, r) if order == "FR" else (r, f)
    q = TEMPLATE["tofu"].format(a=a["question"], b=b["question"])
    item = dict(f)  # F 값 유지: answer / paraphrased_answer / perturbed_answer / domain / author_idx
    item.update(question=q, instruction=q, text_input=q, paraphrased_question="")
    return item, q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    manifest = {"seed": a.seed, "templates": TEMPLATE, "sets": {}}
    for ds, t in [(ds, t) for ds, ts in STEPS.items() for t in ts]:
        name, src = f"{ds}_t{t}", src_paths(ds, t)
        F, R = json.load(open(src["F"])), json.load(open(src["R"]))
        rng = random.Random(f"{a.seed}-{name}")
        pairs = pair_clinc(F, R, rng) if name.startswith("clinc") else pair_tofu(F, R, rng)
        build = build_clinc if name.startswith("clinc") else build_tofu
        for order in ("FR", "RF"):
            path = os.path.join(OUT, f"{name}_{order}.json")
            if os.path.exists(path) and not a.overwrite:
                print(f"[skip] {path} (exists, --overwrite to rebuild)")
            else:
                items = []
                for pid, (fi, ri) in enumerate(pairs):
                    item, q = build(F[fi], R[ri], order, pid)
                    item["cross"] = {"pair_id": pid, "order": order, "template": TEMPLATE[ds], "t": t,
                                     "F": part(F[fi], fi, "F"), "R": part(R[ri], ri, "R")}
                    items.append(item)
                json.dump(items, open(path, "w"), ensure_ascii=False, indent=1)
                print(f"[write] {path}: {len(items)}  ex: {items[0]['question'][:160]!r}")
            manifest["sets"][f"{name}_{order}"] = {"path": path, "md5": md5(path), "n": len(pairs)}
        manifest["sets"][name] = {"ds": ds, "t": t, "F_src": src["F"], "F_md5": md5(src["F"]), "n_F": len(F),
                                  "R_src": src["R"], "R_md5": md5(src["R"]), "n_R": len(R)}
    json.dump(manifest, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
