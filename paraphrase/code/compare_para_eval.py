"""
paraphrase 평가 결과 vs 원본 seed0 결과 (forget / retained choice_accuracy, router_accuracy).
python repro_check/compare_para_eval.py  → repro_check/compare_para_eval.md, .csv
"""
import csv
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "run_results/eval_basemoe_error/llama3.1-8b-instruct")
ORIG = {"tofu": RES + "/repro_tofu_lunar/tofu_moe_fixed_layer16_0",
        "clinc": RES + "/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0"}
TYPES = ["para", "para_ctx", "para_struct", "para_bt", "para_noise"]
SPLITS = ["forget", "retained"]
METRICS = [("choice_accuracy", "acc"), ("router_accuracy", "router")]


def load_run(d):
    out = {}
    for f in glob.glob(os.path.join(d, "lunar_*_t*_*", "forget_*.json")):
        t = int(re.search(r"_t(\d+)_", os.path.basename(os.path.dirname(f))).group(1))
        r = json.load(open(f))
        out[t] = {sp: {m: r[sp]["results"][m] for m, _ in METRICS} for sp in SPLITS if sp in r}
    return out


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def d3(x):
    return "-" if x is None else f"{x:+.3f}"


def main():
    L, rows = ["# Paraphrase 평가 vs 원본 (seed0)\n",
               "acc = choice_accuracy (forget 는 낮을수록 잘 잊음, retained 는 높을수록 좋음). "
               "router = 결과 json router_accuracy (prompt last token).",
               "Δ = para − 원본. 원본: repro_tofu_lunar / repro_clinc_lunar_bs64 seed0.\n"], []
    for ds in ["tofu", "clinc"]:
        orig = load_run(ORIG[ds])
        para = {ty: load_run(f"{RES}/para_eval/{ds}_{ty}/{os.path.basename(ORIG[ds])}") for ty in TYPES}
        steps = sorted(orig)
        L.append(f"\n## {ds.upper()}\n")
        # 요약: 완료된 step 들에서의 평균 Δ
        L.append("### 요약 (완료된 step 평균)\n")
        L.append("| type | 완료 step | forget acc 원본 | forget acc para | Δ | retained acc 원본 | retained acc para | Δ | forget router Δ | retained router Δ |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for ty in TYPES:
            ts = [t for t in steps if t in para[ty]]
            if not ts:
                L.append(f"| {ty} | 0 | | | | | | | | |"); continue
            av = lambda src, sp, m: sum(src[t][sp][m] for t in ts) / len(ts)
            fo, fp = av(orig, "forget", "choice_accuracy"), av(para[ty], "forget", "choice_accuracy")
            ro, rp = av(orig, "retained", "choice_accuracy"), av(para[ty], "retained", "choice_accuracy")
            frd = av(para[ty], "forget", "router_accuracy") - av(orig, "forget", "router_accuracy")
            rrd = av(para[ty], "retained", "router_accuracy") - av(orig, "retained", "router_accuracy")
            rng = f"t{ts[0]}–t{ts[-1]}" if ts == list(range(ts[0], ts[-1] + 1)) else ",".join(map(str, ts))
            L.append(f"| {ty} | {len(ts)} ({rng}) | {f3(fo)} | {f3(fp)} | {d3(fp - fo)} | {f3(ro)} | {f3(rp)} | {d3(rp - ro)} | {d3(frd)} | {d3(rrd)} |")
        for sp in SPLITS:
            for m, mn in METRICS:
                L.append(f"\n### {sp} {mn}\n")
                L.append("| step | 원본 | " + " | ".join(f"{ty} (Δ)" for ty in TYPES) + " |")
                L.append("|---" * (len(TYPES) + 2) + "|")
                for t in steps:
                    o = orig[t][sp][m]
                    cells = []
                    for ty in TYPES:
                        p = para[ty].get(t, {}).get(sp, {}).get(m)
                        cells.append("-" if p is None else f"{p:.3f} ({p - o:+.3f})")
                        rows.append(dict(ds=ds, type=ty, step=t, split=sp, metric=m, orig=o, para=p,
                                         delta=None if p is None else p - o))
                    L.append(f"| t{t} | {o:.3f} | " + " | ".join(cells) + " |")
    out = os.path.join(ROOT, "repro_check/compare_para_eval")
    open(out + ".md", "w").write("\n".join(L) + "\n")
    with open(out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", out + ".md")


if __name__ == "__main__":
    main()
