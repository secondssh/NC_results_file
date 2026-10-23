"""Quantization attack → readme_main_results/quant_attack/"""
import csv
import os

from common import OUT, REPO, RES, copy, d3, f3, load, md_table, render_table, write_csv, write_json

D = f"{OUT}/quant_attack"
Q = f"{REPO}/quant_attack/results"
O3C = f"{REPO}/unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_clinc9_origin"
DOMS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining", "auto_and_commute",
        "utility", "small_talk"]
CS = ["forget", "retained", "mrpc", "rte"]
TS = ["forget", "retained", "real_authors", "world_facts"]
O3SP = {"forget": "testSD", "retained": "testRD", "mrpc": "mrpc", "rte": "rte"}
# 소거 전 base 모델 bf16 평가 (O3_LLAMA31_ClincQA, eval bs 8): domain 별 + mrpc/rte
BASE_STAGE_CSV = f"{REPO}/unlearning_baseline_new/diagnostrics_clinc/baseline_choice_accuracy_by_stage.csv"
BASE_UNITS = load(f"{REPO}/unlearning_baseline_new/run_results/base_eval/llama3.1-8b-instruct/base_units_clinc/base_units_clinc.json")


def rule_json(ds, t):
    if ds == "clinc":
        return (f"{RES}/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0/lunar_llama3.1-8b-instruct_eval_base_"
                f"clincqa_t{t}_{DOMS[t]}/forget_clincqa_t{t}_{DOMS[t]}.json")
    return (f"{RES}/repro_tofu_lunar/tofu_moe_fixed_layer16_0/lunar_llama3.1-8b-instruct_eval_base_tofu_t{t}_AG{t+1:02d}/"
            f"forget_tofu_t{t}_AG{t+1:02d}.json")


def words(gen):
    return sum(len(str(g[1]).split()) for g in gen) / max(len(gen), 1)


def ev(p):
    """eval json → {split: {acc, router, n, words}}"""
    d = load(p)
    if d is None:
        return None
    out = {}
    for sp, v in d.items():
        if isinstance(v, dict) and "results" in v:
            r = v["results"]
            gen = r.get("generated_text") or []
            out[sp] = {"acc": r.get("choice_accuracy"), "router": r.get("router_accuracy"),
                       "n": r.get("total_generated", len(gen)), "words": words(gen) if gen else None}
    return out


def o3(fn):
    out = {}
    for sp, src in O3SP.items():
        d = load(fn(src))
        if d is not None:
            res = d.get("results") or []
            out[sp] = {"acc": d["exact_acc"] / 100, "contains": d["contains_acc"] / 100, "n": d["len"],
                       "words": (sum(len(str(x.get("raw_output", "")).split()) for x in res) / len(res)) if res else None,
                       "ood_w": (sum(float(x.get("ood_weight") or 0) for x in res) / len(res)) if res else None}
    return out or None


def base_t0(p):
    """base 는 ckpt 무관 → t8 전체 pool 평가에서 t0 SD(work)/RD(나머지 8 domain SD + meta RD) 로 재분할"""
    d = load(p)
    if d is None:
        return None
    out = {}
    for sp in ("mrpc", "rte"):
        if sp in d:
            out[sp] = {"acc": d[sp]["results"]["choice_accuracy"]}
    fg, rt = d["forget"]["results"], d["retained"]["results"]
    fail = {}
    for g in fg["failure_text"]:
        fail[g[0]] = fail.get(g[0], 0) + 1
    seen = {}
    f_hit = f_n = r_hit = r_n = 0
    for g in fg["generated_text"]:
        k = g[0]
        hit = seen.get(k, 0) < fail.get(k, 0)
        seen[k] = seen.get(k, 0) + 1
        if g[3] == "work":
            f_n += 1
            f_hit += hit
        else:
            r_n += 1
            r_hit += hit
    r_hit += rt["choice_correct"]
    r_n += rt["total_generated"]
    out["forget"] = {"acc": f_hit / f_n, "n": f_n}
    out["retained"] = {"acc": r_hit / r_n, "n": r_n}
    return out


def main():
    copy(Q, f"{D}/results/quant_attack_results")
    for f in ["fake_quant.py", "quant_utils.py", "eval_quant_clinc.py", "eval_quant_tofu.py", "eval_quant_o3_clinc.py",
              "eval_quant_o3_tofu.py", "run_quant_clinc.sh", "run_quant_tofu.sh", "run_quant_o3_clinc.sh",
              "run_quant_t0_clinc.sh", "run_quant_t0_gd.sh", "run_quant_t4_rule.sh", "run_quant_tofu_t0_t10.sh", "weight_delta_nf4.py",
              "summarize_quant_table.py"]:
        copy(f"{REPO}/quant_attack/{f}", f"{D}/code/{f}")
    for d in ("bf16", "backbone", "all"):
        pass
    O3_BF = lambda t: o3(lambda s: f"{O3C}/test_seed0_{DOMS[t]}_{s}.json")  # noqa: E731
    # base bf16 = unlearning_baseline_new/diagnostrics_clinc/baseline_choice_accuracy_by_stage.csv (stage 별 BASE 행)
    base_stage = {}
    for r in csv.DictReader(open(BASE_STAGE_CSV)):
        if r["method"] == "BASE":
            base_stage.setdefault(int(r["stage"]), {})[r["split"]] = {"acc": float(r["choice_accuracy"]),
                                                                      "n": int(r["n_eval"])}
    base_bf_t8 = base_stage[8]
    base_bf_t8["forget"]["words"] = (ev(f"{RES}/para_eval/base/clinc_orig.json") or {}).get("forget", {}).get("words")
    R = {
        ("clinc", "RULE t0"): {"bf16": ev(rule_json("clinc", 0)), "backbone": ev(f"{Q}/clinc_t0_bs64_fake/ours_backbone.json"),
                               "all": ev(f"{Q}/clinc_t0_bs64_fake/ours_all.json")},
        ("clinc", "RULE t4"): {"bf16": ev(rule_json("clinc", 4)), "backbone": ev(f"{Q}/clinc_t4_bs64_fake/ours_backbone.json"),
                               "all": ev(f"{Q}/clinc_t4_bs64_fake/ours_all.json")},
        ("clinc", "RULE t8"): {"bf16": ev(rule_json("clinc", 8)), "backbone": ev(f"{Q}/clinc_t8_bs64_fake/ours_backbone.json"),
                               "all": ev(f"{Q}/clinc_t8_bs64_fake/ours_all.json")},
        ("clinc", "O3 t0"): {"bf16": O3_BF(0), "backbone": o3(lambda s: f"{Q}/o3_clinc_t0_oodrecent/backbone_{s}.json"),
                             "all": o3(lambda s: f"{Q}/o3_clinc_t0_oodrecent/all_{s}.json")},
        ("clinc", "GD t0"): {"bf16": ev(f"{Q}/clinc_t0_gd_bs8_fake/ga_gd_none.json"),
                             "backbone": ev(f"{Q}/clinc_t0_gd_bs8_fake/ga_gd_backbone.json"), "all": None},
        ("clinc", "base t0"): {"bf16": base_stage[0],
                               "backbone": base_t0(f"{Q}/clinc_t8_bs64_fake/base_backbone.json"), "all": None},
        ("clinc", "base t8"): {"bf16": base_bf_t8, "backbone": ev(f"{Q}/clinc_t8_bs64_fake/base_backbone.json"), "all": None},
        ("tofu", "RULE t0"): {"bf16": ev(rule_json("tofu", 0)), "backbone": ev(f"{Q}/tofu_t0_fake/ours_backbone.json"),
                              "all": ev(f"{Q}/tofu_t0_fake/ours_all.json")},
        ("tofu", "RULE t10"): {"bf16": ev(rule_json("tofu", 10)), "backbone": ev(f"{Q}/tofu_t10_fake/ours_backbone.json"),
                               "all": ev(f"{Q}/tofu_t10_fake/ours_all.json")},
    }
    write_json(f"{D}/quant_summary.json", {f"{k[0]}|{k[1]}": v for k, v in R.items()})

    md_parts = []
    for ds, sps in (("clinc", CS), ("tofu", TS)):
        header = ["방법", "조건"] + sps + ["forget 생성 단어수"]
        rows, hl = [], []
        for (d2, meth), conds in R.items():
            if d2 != ds:
                continue
            ref = conds["bf16"] or {}
            for ci, (c, lab) in enumerate([("bf16", "bf16 (원래)"), ("backbone", "백본 4bit"),
                                           ("all", "백본+MoE 4bit" if "RULE" in meth else "백본+LoRA 4bit")]):
                v = conds[c]
                if v is None:
                    continue
                if meth.startswith("GD") and c == "backbone":
                    lab = "전체 4bit (full-FT)"
                if meth.startswith("base") and c == "backbone":
                    lab = "전체 4bit"
                cells = [(f3(v.get(sp, {}).get("acc")) if c == "bf16" else
                          d3(v.get(sp, {}).get("acc"), ref.get(sp, {}).get("acc"))) for sp in sps]
                w = v.get("forget", {}).get("words")
                rows.append([meth if ci == 0 else "", lab] + cells + [f"{w:.1f}" if w else "-"])
                if c == "bf16":
                    hl.append(len(rows) - 1)
        write_csv(f"{D}/tables/quant_{ds}.csv", header, rows)
        note = ("NF4 4bit fake quant (양자화 → bf16 복원), seed 0. 괄호 = bf16 대비 Δ. forget 은 높아지면 지식이 되살아난 것.\n"
                "RULE/GD/base = choice_accuracy, O3 = exact_acc (OOD detector clincqa_recent). "
                "base bf16 = baseline_choice_accuracy_by_stage.csv (BASE, stage 0 / 8). base 4bit t0 = t8 전체 pool 평가를 t0 SD/RD 로 재분할 (base 는 ckpt 무관).")
        if ds == "tofu":
            note = "NF4 4bit fake quant, seed 0. 괄호 = bf16 대비 Δ. choice_accuracy. TOFU t19 / O3 TOFU / base TOFU 양자화는 미실행."
        render_table(f"{D}/tables/quant_{ds}.png", header, rows, title=f"Quantization attack — {ds.upper()}",
                     note=note, highlight_rows=hl, col_max=30)
        md_parts.append(f"### {ds.upper()}\n\n" + md_table(header, rows))

    # O3 CLINC bf16 step 별 (왜 t0 를 골랐는지)
    o3rows = []
    for t in range(9):
        v = O3_BF(t) or {}
        r = load(rule_json("clinc", t))
        o3rows.append([f"t{t}_{DOMS[t]}"] + [f3(v.get(sp, {}).get("acc")) for sp in CS] +
                      [f3(r[sp]["results"]["choice_accuracy"]) if r and sp in r else "-" for sp in ("forget", "retained")])
    h = ["step", "O3 forget", "O3 retained", "O3 mrpc", "O3 rte", "RULE forget", "RULE retained"]
    write_csv(f"{D}/tables/o3_vs_rule_clinc_bf16_steps.csv", h, o3rows)
    render_table(f"{D}/tables/o3_vs_rule_clinc_bf16_steps.png", h, o3rows,
                 title="CLINC bf16 step 별 — O3 (exact_acc) vs RULE (choice_acc), seed0",
                 note="O3 는 t1 부터 forget 이 다시 올라감 (t2 이후 0.68+) → 양자화 비교는 O3 가 잊은 상태인 t0 에서 수행")
    md_parts.append("### O3 vs RULE CLINC bf16 step 별\n\n" + md_table(h, o3rows))

    # 예전 결과 (10/16, bs1024 ckpt, bnb 실제 4bit, eval bs8) — 기록용
    L = f"{Q}/clinc_t8"
    leg = []
    for tag, lab in [("base_bits0", "base bf16"), ("base_bits4", "base bnb 4bit"), ("ours_bits0", "RULE(bs1024 ckpt) bf16"),
                     ("ours_bits4", "RULE(bs1024 ckpt) 백본 bnb 4bit"), ("ga_bits4", "GA bnb 4bit"),
                     ("npo_kl_bits4", "NPO+KL bnb 4bit")]:
        v = ev(f"{L}/{tag}.json")
        if v:
            leg.append([lab] + [f3(v.get(sp, {}).get("acc")) for sp in CS] + [f"{v['forget']['words']:.1f}" if v["forget"].get("words") else "-"])
    h = ["모델/조건 (legacy)"] + CS + ["forget 생성 단어수"]
    write_csv(f"{D}/tables/legacy_clinc_t8_bs1024.csv", h, leg)
    render_table(f"{D}/tables/legacy_clinc_t8_bs1024.png", h, leg,
                 title="[legacy] CLINC t8 — 10/16 SLURM 실행 (bs1024 ckpt, bnb 실제 4bit, eval bs8)",
                 note="논문 설정(bs64)과 다른 예전 ckpt 결과. 본문 비교에 쓰지 않음, 기록용.")
    md_parts.append("### [legacy] CLINC t8 bs1024 ckpt\n\n" + md_table(h, leg))

    wd = load(f"{Q}/weight_delta_gd_t0.json")
    if wd:
        s = wd["summary"]
        h = ["지표", "값"]
        rows = [["n_tensors", s["n_tensors"]], ["same_frac (NF4 값이 base 와 완전히 같은 원소)", f3(s["same_frac"])],
                ["changed_frac (bf16 에서 바뀐 원소)", f3(s["changed_frac"])], ["rel_delta ‖ΔW‖/‖W‖", f"{s['rel_delta']:.4f}"],
                ["keep_ratio ‖Q(Wu)-Q(Wb)‖/‖ΔW‖", f3(s["keep_ratio"])], ["median abs(ΔW) / NF4 눈금 절반", f"{s['median_bin_ratio']:.4f}"]]
        write_csv(f"{D}/tables/gd_t0_weight_delta_nf4.csv", h, rows)
        render_table(f"{D}/tables/gd_t0_weight_delta_nf4.png", h, rows, title="GD t0 가중치 변화 vs NF4 눈금 (CPU 분석)")
        md_parts.append("### GD t0 weight delta vs NF4\n\n" + md_table(h, rows))
    open(f"{D}/tables/all_tables.md", "w").write("\n\n".join(md_parts) + "\n")
    print("quant done")


if __name__ == "__main__":
    main()
