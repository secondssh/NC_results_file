"""Paraphrase 평가 → readme_main_results/paraphrase/"""
import glob
import json
import os
import re

from common import MC, OUT, REPO, RES, copy, md_to_csvs, render_table, write_csv, write_json

D = f"{OUT}/paraphrase"
PE = f"{RES}/para_eval"
ORIG = {"tofu": f"{RES}/repro_tofu_lunar/tofu_moe_fixed_layer16_0",
        "clinc": f"{RES}/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0"}
TYPES = ["orig", "para", "para_ctx", "para_struct", "para_bt", "para_noise"]
KEEP = ("choice_accuracy", "router_accuracy", "label_contains_accuracy", "gt_loss_per_token", "probs",
        "rouge_l_recall", "token_f1", "exact_match", "total_generated")


def metrics_of(f):
    r = json.load(open(f))
    return {sp: {k: v["results"][k] for k in KEEP if k in v["results"]}
            for sp, v in r.items() if isinstance(v, dict) and "results" in v}


def main():
    copy(PE, f"{D}/results/para_eval", ignore=("_locks",))
    for f in ["compare_para_eval.md", "compare_para_eval.csv", "base_para_steps.md", "base_para_steps.csv",
              "para_eval_steps.png"]:
        copy(f"{MC}/repro_check/{f}", f"{D}/results/{f}")
    copy(f"{REPO}/data_new/paraphrase/maps", f"{D}/paraphrase_data/maps")
    copy(f"{REPO}/data_new/paraphrase/logs", f"{D}/paraphrase_data/gen_logs", ignore=("*.bak",))
    for f in ["compare_para_eval.py", "base_para_steps.py", "plot_para_eval.py", "run_para_eval.sh",
              "para_eval_supervisor_v2.sh", "run_base_para_eval.sh"]:
        copy(f"{MC}/repro_check/{f}", f"{D}/code/{f}")
    copy(f"{REPO}/data_new/paraphrase/make_para.py", f"{D}/code/make_para.py")

    # 모든 run 의 scalar 지표 (RULE 원본 seed0 + 5 유형, base 6 세트)
    allm, rows = {}, []
    for ds in ("tofu", "clinc"):
        for ty in TYPES:
            root = ORIG[ds] if ty == "orig" else f"{PE}/{ds}_{ty}/" + os.path.basename(ORIG[ds])
            for f in sorted(glob.glob(f"{root}/lunar_*_t*_*/forget_*.json")):
                t = int(re.search(r"_t(\d+)_", os.path.basename(os.path.dirname(f))).group(1))
                m = metrics_of(f)
                allm.setdefault(ds, {}).setdefault(ty, {})[f"t{t}"] = m
                for sp in ("forget", "retained"):
                    if sp in m:
                        rows.append([ds, "RULE", ty, t, sp, m[sp].get("choice_accuracy"), m[sp].get("router_accuracy")])
            b = f"{PE}/base/{ds}_{ty}.json"
            if os.path.isfile(b):
                m = metrics_of(b)
                allm.setdefault(ds, {}).setdefault(ty, {})["base_fullpool"] = m
                for sp in ("forget", "retained"):
                    rows.append([ds, "base(full pool)", ty, "", sp, m[sp].get("choice_accuracy"), None])
    write_json(f"{D}/all_metrics.json", allm)
    rows.sort(key=lambda r: (r[0], r[1], TYPES.index(r[2]), r[3] if r[3] != "" else -1, r[4]))
    write_csv(f"{D}/all_metrics.csv", ["dataset", "model", "type", "step", "split", "choice_accuracy",
                                       "router_accuracy"], rows)

    tabs = md_to_csvs(f"{MC}/repro_check/compare_para_eval.md", f"{D}/tables", "rule_vs_orig")
    tabs += md_to_csvs(f"{MC}/repro_check/base_para_steps.md", f"{D}/tables", "base_vs_rule")
    for p, t in tabs:
        name = os.path.basename(p)[:-4]
        title = {"rule_vs_orig": "RULE paraphrase vs 원문 (seed0) — ", "base_vs_rule": "base(unlearning 전) / RULE — "}[
            name.split("__")[0]] + t["head"]
        wide = len(t["header"]) > 7
        render_table(f"{D}/tables/{name}.png", t["header"], t["rows"], title=title,
                     font=8 if wide else 9, col_max=22 if wide else 40)
    print("para done")


if __name__ == "__main__":
    main()
