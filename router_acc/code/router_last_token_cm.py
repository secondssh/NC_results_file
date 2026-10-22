"""
Prompt-last-token router acc + forget/retain 혼합 confusion matrix (GPU 불필요).

결과 json 의 generated_text[i][4] = router_info
  {"router_pred_expert", "router_target_expert", "router_layer_preds"}
는 eval 시 compute_router_accuracy_BD 가 prompt 마지막 토큰 activation 을 gate 에 넣어 얻은 값
(= 결과 json 의 router_accuracy 와 같은 기준). target: retained=0, forget=domain 번호(1..).

python repro_check/router_last_token_cm.py            # repro_tofu_lunar, repro_clinc_lunar_bs64 (seed 0,1,2)
출력: repro_check/router_metrics/last_token_{tofu,clinc}.{json,md}, cm_csv/
"""
import glob
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "run_results/eval_basemoe_error/llama3.1-8b-instruct")
OUT = os.path.join(ROOT, "repro_check/router_metrics")
RUNS = {
    "tofu": os.path.join(RES, "repro_tofu_lunar/tofu_moe_fixed_layer16_{s}"),
    "clinc": os.path.join(RES, "repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_{s}"),
}
SEEDS = [0, 1, 2]
SPLITS = ["forget", "retained"]


def step_dirs(run_dir):
    out = []
    for d in glob.glob(os.path.join(run_dir, "lunar_*_t*_*")):
        m = re.search(r"_t(\d+)_(.+)$", d)
        out.append((int(m.group(1)), m.group(2), d))
    return sorted(out)


def load_pairs(step_dir):
    f = glob.glob(os.path.join(step_dir, "forget_*.json"))[0]
    d = json.load(open(f))
    pairs, saved = {}, {}
    for sp in d:
        r = d[sp]["results"]
        pairs[sp] = [(x[4]["router_target_expert"], x[4]["router_pred_expert"]) for x in r["generated_text"]]
        saved[sp] = r.get("router_accuracy")
    return pairs, saved


def analyze_step(t, pairs, saved):
    n_exp = t + 2  # expert 0 = retain, 1..t+1 = forget domains
    cm = np.zeros((n_exp, n_exp), dtype=np.int64)  # row = target, col = pred
    row = {}
    for sp, pr in pairs.items():
        tg = np.array([p[0] for p in pr])
        pd = np.array([p[1] for p in pr])
        acc = float((tg == pd).mean())
        if saved.get(sp) is not None and abs(acc - saved[sp]) > 1e-9:
            print(f"  [warn] t{t} {sp}: recomputed {acc} != saved {saved[sp]}", file=sys.stderr)
        row[f"{sp}_acc"] = acc
        if sp in SPLITS:
            for a, b in zip(tg, pd):
                cm[a, b] += 1
    n_f = cm[1:].sum()
    n_r = cm[0].sum()
    row["mixed_acc"] = float(np.trace(cm) / cm.sum())
    # forget vs retain 이진 (forget expert 어느 것이든 → forget 으로 간주)
    row["bin"] = {
        "F->F": int(cm[1:, 1:].sum()), "F->R": int(cm[1:, 0].sum()),
        "R->F": int(cm[0, 1:].sum()), "R->R": int(cm[0, 0]),
    }
    row["forget_bin_acc"] = row["bin"]["F->F"] / n_f
    row["retain_bin_acc"] = row["bin"]["R->R"] / n_r
    row["forget_wrong_forget_expert"] = row["bin"]["F->F"] - int(np.trace(cm[1:, 1:]))
    row["n_forget"], row["n_retain"] = int(n_f), int(n_r)
    row["cm"] = cm.tolist()
    return row


def fmt(x):
    return "-" if x is None else f"{x:.4f}"


def main():
    os.makedirs(os.path.join(OUT, "cm_csv"), exist_ok=True)
    for ds, pat in RUNS.items():
        allres = {}
        for s in SEEDS:
            rd = pat.format(s=s)
            allres[s] = {}
            for t, dom, sd in step_dirs(rd):
                pairs, saved = load_pairs(sd)
                r = analyze_step(t, pairs, saved)
                r["domain"] = dom
                allres[s][t] = r
                labels = ["R(e0)"] + [f"F{k}" for k in range(1, t + 2)]
                with open(os.path.join(OUT, "cm_csv", f"{ds}_s{s}_t{t}_{dom}.csv"), "w") as f:
                    f.write("target\\pred," + ",".join(f"e{k}" for k in range(t + 2)) + "\n")
                    for lab, rr in zip(labels, r["cm"]):
                        f.write(lab + "," + ",".join(map(str, rr)) + "\n")
        json.dump({"dataset": ds, "runs": {s: pat.format(s=s) for s in SEEDS}, "results": allres},
                  open(os.path.join(OUT, f"last_token_{ds}.json"), "w"), indent=1)

        steps = sorted(allres[SEEDS[0]])
        util = [k[:-4] for k in allres[SEEDS[0]][steps[0]] if k.endswith("_acc")
                and k[:-4] not in SPLITS + ["mixed", "forget_bin", "retain_bin"]]
        L = [f"# Prompt-last-token router acc — {ds}\n",
             "기준: eval 시 저장된 router_info (prompt 마지막 토큰 activation → layer16 gate argmax, "
             "결과 json `router_accuracy`와 동일). target: retained=e0, forget=도메인 expert.\n",
             "- mixed = forget+retain 합쳐서 정확히 맞춘 비율 (micro)",
             "- F-bin = forget이 *아무* forget expert로 간 비율, R-bin = retain이 e0로 간 비율 (= retained acc)\n"]
        for metric in ["forget_acc", "retained_acc", "mixed_acc", "forget_bin_acc"] + [u + "_acc" for u in util]:
            L.append(f"\n## {metric}\n")
            L.append("| step | " + " | ".join(f"s{s}" for s in SEEDS) + " | mean |")
            L.append("|---" * (len(SEEDS) + 2) + "|")
            for t in steps:
                v = [allres[s][t].get(metric) for s in SEEDS if t in allres[s]]
                L.append(f"| t{t}_{allres[SEEDS[0]][t]['domain']} | " + " | ".join(fmt(x) for x in v)
                         + f" | {fmt(float(np.mean(v)))} |")
            v = [np.mean([allres[s][t][metric] for t in steps]) for s in SEEDS]
            L.append("| **avg** | " + " | ".join(fmt(x) for x in v) + f" | {fmt(float(np.mean(v)))} |")

        L.append("\n## forget/retain 이진 confusion (seed 0+1+2 합)\n")
        L.append("| step | F→F(맞는 e) | F→F(다른 forget e) | F→R(e0) | R→R(e0) | R→F |")
        L.append("|---|---|---|---|---|---|")
        for t in steps:
            b = {k: sum(allres[s][t]["bin"][k] for s in SEEDS) for k in ["F->F", "F->R", "R->F", "R->R"]}
            wr = sum(allres[s][t]["forget_wrong_forget_expert"] for s in SEEDS)
            L.append(f"| t{t} | {b['F->F'] - wr} | {wr} | {b['F->R']} | {b['R->R']} | {b['R->F']} |")

        for t in [steps[-1]]:
            L.append(f"\n## 전체 confusion matrix t{t} (seed0, row=target, col=pred)\n")
            cm = allres[0][t]["cm"]
            L.append("| target\\pred | " + " | ".join(f"e{k}" for k in range(len(cm))) + " |")
            L.append("|---" * (len(cm) + 1) + "|")
            for k, rr in enumerate(cm):
                L.append(f"| {'R(e0)' if k == 0 else f'F{k}'} | " + " | ".join(map(str, rr)) + " |")
        L.append(f"\n모든 seed/step 행렬: `repro_check/router_metrics/cm_csv/{ds}_s*_t*.csv`")
        open(os.path.join(OUT, f"last_token_{ds}.md"), "w").write("\n".join(L) + "\n")
        print("wrote", os.path.join(OUT, f"last_token_{ds}.md"))


if __name__ == "__main__":
    main()
