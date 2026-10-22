"""
router_token_eval.py 결과 → step × seed 표 (md) + confusion matrix csv.
python repro_check/summarize_router_token.py   → repro_check/router_metrics/token_{ds}.md, token_cm_csv/
"""
import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK = os.path.join(ROOT, "repro_check/router_metrics/token")
OUT = os.path.join(ROOT, "repro_check/router_metrics")
SEEDS = [0, 1, 2]
METRICS = [
    ("prompt_tok_acc", "prompt token (전체, 템플릿 포함)"),
    ("prompt_content_tok_acc", "prompt token (템플릿 제외 내용 토큰)"),
    ("prompt_last_acc", "prompt last token (생성 시점 실제 routing)"),
    ("saved_router_accuracy", "prompt last token (결과 json router_accuracy, 논문 지표)"),
    ("gen_tok_acc", "generated token"),
    ("gen_majority_acc", "generated token 샘플별 다수결"),
]
CMS = [("last", "prompt last token"), ("prompt_tok", "prompt token"),
       ("prompt_content_tok", "prompt 내용 토큰"), ("gen_tok", "generated token")]


def load(ds):
    res = {}
    for s in SEEDS:
        for f in glob.glob(os.path.join(TOK, f"{ds}_s{s}", "t*.json")):
            if f.endswith("_util.json"):
                continue
            d = json.load(open(f))
            res[(s, d["t"])] = d
    return res


def fmt(x):
    return "-" if x is None else f"{x:.4f}"


def mixed_saved(d):
    f, r = d["splits"]["forget"], d["splits"]["retained"]
    return (f["saved_router_accuracy"] * f["n"] + r["saved_router_accuracy"] * r["n"]) / (f["n"] + r["n"])


def get(d, sp, m):
    if m == "saved_router_accuracy" and sp == "mixed":
        return mixed_saved(d)
    return d["splits"][sp].get(m)


UTIL_METRICS = [
    ("prompt_last_acc", "prompt last token (생성 시점 실제 routing)"),
    ("saved_router_accuracy", "prompt last token (결과 json router_accuracy)"),
    ("prompt_tok_acc", "prompt token (전체)"),
    ("prompt_content_tok_acc", "prompt token (템플릿 제외)"),
    ("gen_tok_acc_sample_mean", "generated token (샘플 평균)"),
    ("gen_majority_acc", "generated token 다수결"),
]


def util_section(ds):
    """router_token_eval.py --util 결과 (t*_util.json). target = e0 (원래 모델)."""
    res = {}
    for s in SEEDS:
        for f in glob.glob(os.path.join(TOK, f"{ds}_s{s}", "t*_util.json")):
            d = json.load(open(f))
            res[(s, d["t"])] = d
    if not res:
        return []
    steps = sorted({t for _, t in res})
    sps = list(next(iter(res.values()))["splits"])
    L = [f"\n## Utility split router acc ({', '.join(sps)}) — target = e0 (원래 모델)\n",
         "utility 문항은 어느 forget 도메인에도 속하지 않으므로 e0 로 가야 정답. "
         "완료: " + " ".join(f"s{s}:{sum(1 for (ss, _) in res if ss == s)}" for s in SEEDS) + "\n"]
    if ds == "tofu":
        L.append("TOFU 의 `router_accuracy`(결과 json)는 left padding + position_ids 없음 때문에 실제 routing 과 다름 → "
                 "`prompt last token (생성 시점)` 을 쓸 것.\n")
    L.append("### 요약 (step 평균, seed 평균)\n")
    L.append("| metric | " + " | ".join(sps) + " |")
    L.append("|---" * (len(sps) + 1) + "|")
    for m, desc in UTIL_METRICS:
        row = []
        for sp in sps:
            v = [d["splits"][sp].get(m) for d in res.values() if d["splits"][sp].get(m) is not None]
            row.append(fmt(float(np.mean(v))) if v else "-")
        L.append(f"| {desc} | " + " | ".join(row) + " |")
    for sp in sps:
        for m, desc in [UTIL_METRICS[0], UTIL_METRICS[3], UTIL_METRICS[4]]:
            L.append(f"\n### {sp} — {desc} (`{m}`)\n")
            L.append("| step | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean | (saved mean) |" if m == "prompt_last_acc"
                     else "| step | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean |")
            L.append("|---" * (len(SEEDS) + (3 if m == "prompt_last_acc" else 2)) + "|")
            for t in steps:
                v = [res[(s, t)]["splits"][sp].get(m) if (s, t) in res else None for s in SEEDS]
                vv = [x for x in v if x is not None]
                row = f"| t{t}_{next(d for (ss, tt), d in res.items() if tt == t)['domain']} | " + \
                      " | ".join(fmt(x) for x in v) + f" | {fmt(float(np.mean(vv))) if vv else '-'} |"
                if m == "prompt_last_acc":
                    sv = [res[(s, t)]["splits"][sp]["saved_router_accuracy"] for s in SEEDS if (s, t) in res]
                    row += f" {fmt(float(np.mean(sv)))} |"
                L.append(row)
    # utility 문항이 어느 expert 로 갔는지 (prompt last token, seed 합, 마지막 step)
    last = steps[-1]
    L.append(f"\n### 마지막 step t{last}: utility 문항의 expert 분포 (prompt last token, seed 합)\n")
    n_exp = len(next(iter(res.values()))["splits"][sps[0]]["cm"]["last"])
    L.append("| split | " + " | ".join(f"e{k}" for k in range(n_exp)) + " |")
    L.append("|---" * (n_exp + 1) + "|")
    for sp in sps:
        row = sum(np.array(res[(s, last)]["splits"][sp]["cm"]["last"])[0] for s in SEEDS if (s, last) in res)
        L.append(f"| {sp} | " + " | ".join(str(int(x)) for x in row) + " |")
    return L


def main():
    os.makedirs(os.path.join(OUT, "token_cm_csv"), exist_ok=True)
    for ds in ["clinc", "tofu"]:
        res = load(ds)
        if not res:
            continue
        steps = sorted({t for _, t in res})
        dom = {t: d["domain"] for (_, t), d in res.items()}
        L = [f"# Token-level router acc — {ds}\n",
             "생성문은 결과 json 에 저장된 것을 그대로 teacher forcing (재생성 X), layer16 gate argmax.",
             "target: retained → e0, forget → 해당 domain expert. mixed = forget+retain 합친 micro.",
             "검증: 실제 generate 중 기록한 routing 과 prompt 99.7%+ / generated 99.1%+ 토큰 일치, 재생성문 100% 동일.\n",
             "| 완료 | " + " ".join(f"s{s}:{sum(1 for (ss, _) in res if ss == s)}" for s in SEEDS) + " |\n"]

        L.append("## 요약 (step 평균, seed 평균)\n")
        L.append("| metric | forget | retained | mixed |")
        L.append("|---|---|---|---|")
        for m, desc in METRICS:
            row = []
            for sp in ["forget", "retained", "mixed"]:
                v = [get(d, sp, m) for d in res.values() if get(d, sp, m) is not None]
                row.append(fmt(float(np.mean(v))) if v else "-")
            L.append(f"| {desc} | " + " | ".join(row) + " |")

        for sp in ["forget", "retained", "mixed"]:
            L.append(f"\n## {sp}\n")
            for m, desc in METRICS:
                L.append(f"\n### {sp} — {desc} (`{m}`)\n")
                L.append("| step | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean |")
                L.append("|---" * (len(SEEDS) + 2) + "|")
                per_seed = {s: [] for s in SEEDS}
                for t in steps:
                    v = []
                    for s in SEEDS:
                        x = get(res[(s, t)], sp, m) if (s, t) in res else None
                        v.append(x)
                        if x is not None:
                            per_seed[s].append(x)
                    vv = [x for x in v if x is not None]
                    L.append(f"| t{t}_{dom[t]} | " + " | ".join(fmt(x) for x in v)
                             + f" | {fmt(float(np.mean(vv))) if vv else '-'} |")
                avg = [float(np.mean(per_seed[s])) if per_seed[s] else None for s in SEEDS]
                aa = [x for x in avg if x is not None]
                L.append("| **avg** | " + " | ".join(fmt(x) for x in avg) + f" | {fmt(float(np.mean(aa))) if aa else '-'} |")

        # forget/retain 이진 (seed 합): 생성 시점 last token, generated token
        for key, desc in [("last", "prompt last token (샘플 수)"), ("gen_tok", "generated token (토큰 수)")]:
            L.append(f"\n## forget/retain 이진 confusion — {desc}, seed 합\n")
            L.append("| step | F→맞는 e | F→다른 forget e | F→e0 | R→e0 | R→forget e |")
            L.append("|---|---|---|---|---|---|")
            for t in steps:
                cm = sum(np.array(res[(s, t)]["splits"]["mixed"]["cm"][key]) for s in SEEDS if (s, t) in res)
                ff = cm[1:, 1:].sum(); diag = np.trace(cm[1:, 1:])
                L.append(f"| t{t} | {diag} | {ff - diag} | {cm[1:, 0].sum()} | {cm[0, 0]} | {cm[0, 1:].sum()} |")

        last = steps[-1]
        if (0, last) in res:
            for key, desc in CMS:
                cm = res[(0, last)]["splits"]["mixed"]["cm"][key]
                L.append(f"\n## 전체 confusion matrix t{last} seed0 — {desc} (row=target, col=pred)\n")
                L.append("| target\\pred | " + " | ".join(f"e{k}" for k in range(len(cm))) + " |")
                L.append("|---" * (len(cm) + 1) + "|")
                for k, rr in enumerate(cm):
                    L.append(f"| {'R(e0)' if k == 0 else f'F{k}'} | " + " | ".join(map(str, rr)) + " |")
        L += util_section(ds)
        L.append(f"\n모든 seed/step/종류 행렬: `repro_check/router_metrics/token_cm_csv/{ds}_s*_t*_{{last,prompt_tok,prompt_content_tok,prompt_maj,gen_tok,gen_maj}}.csv`")

        for (s, t), d in res.items():
            for key, cm in d["splits"]["mixed"]["cm"].items():
                with open(os.path.join(OUT, "token_cm_csv", f"{ds}_s{s}_t{t}_{d['domain']}_{key}.csv"), "w") as f:
                    f.write("target\\pred," + ",".join(f"e{k}" for k in range(len(cm))) + "\n")
                    for k, rr in enumerate(cm):
                        f.write(("R(e0)" if k == 0 else f"F{k}") + "," + ",".join(map(str, rr)) + "\n")
        open(os.path.join(OUT, f"token_{ds}.md"), "w").write("\n".join(L) + "\n")
        print("wrote", os.path.join(OUT, f"token_{ds}.md"))


if __name__ == "__main__":
    main()
