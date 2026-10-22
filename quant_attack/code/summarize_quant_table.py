"""Quantization attack 최종 표: RULE / O3 / base × CLINC t8 / TOFU t19, 열 = bf16(원래) | backbone 4bit | 전체 4bit.

  RULE CLINC : 원래 = modify_code/.../repro_clinc_lunar_bs64 seed0 t8, 양자화 = results/clinc_t8_bs64_fake/ours_{mode}.json
  RULE TOFU  : 원래 = modify_code/.../repro_tofu_lunar seed0 t19,      양자화 = results/tofu_t19_fake/ours_{mode}.json
  base       : 소거 전 모델 (bf16 = *_none.json 이 있으면 그것, CLINC 는 없으면 예전 results/clinc_t8/base_bits0.json)
  O3 CLINC   : 원래 = unlearn_recent_o3/.../test_seed0_small_talk_{split}.json, 양자화 = results/o3_clinc_t8_ood{OOD}/{mode}_{split}.json
  O3 TOFU    : 원래 = llama31_8b_O3/seed_0/stage_19/ptofu.json,           양자화 = results/o3_tofu_stage19/{mode}/ptofu.json

지표: RULE/base = choice_accuracy (각 eval 의 기본 채점), O3 CLINC = exact_acc (eval_o3_clinc 채점),
O3 TOFU = choice_accuracy (eval_ptofu). O3 TOFU 는 forget=현재 저자 40문항, retained 200문항으로 RULE 과 범위가 다름.

python quant_attack/summarize_quant_table.py [--o3_ood recent|9]  → results/quant_table.md
"""
import argparse
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                                    # ScienceQA_expriment
ROOT = os.path.dirname(REPO)
RES = os.path.join(REPO, "modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct")
O3C = os.path.join(REPO, "unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_clinc9_origin")
O3T = os.path.join(ROOT, "PTOFU_experiment/files/results/unlearn_ptofu/llama31_8b_O3/seed_0/stage_19/ptofu.json")
MODES = ["bf16", "backbone", "all"]
SPLITS = {"clinc": ["forget", "retained", "mrpc", "rte"],
          "tofu": ["forget", "retained", "real_authors", "world_facts"]}
O3C_SPLIT = {"forget": "testSD", "retained": "testRD", "mrpc": "mrpc", "rte": "rte"}


def _load(p):
    return json.load(open(p)) if p and os.path.isfile(p) else None


def eval_json_acc(p):
    d = _load(p)
    if d is None:
        return None
    return {sp: v["results"].get("choice_accuracy") for sp, v in d.items()
            if sp != "_quant_meta" and isinstance(v, dict) and "results" in v}


def o3_clinc_acc(p_of_split):
    out = {}
    for sp, src in O3C_SPLIT.items():
        d = _load(p_of_split(src))
        out[sp] = None if d is None else d["exact_acc"] / 100
    return out if any(v is not None for v in out.values()) else None


def rows(o3_ood):
    q = os.path.join(HERE, "results")
    R = {}
    R[("clinc", "RULE")] = {
        "bf16": eval_json_acc(f"{RES}/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0/"
                              "lunar_llama3.1-8b-instruct_eval_base_clincqa_t8_small_talk/forget_clincqa_t8_small_talk.json"),
        "backbone": eval_json_acc(f"{q}/clinc_t8_bs64_fake/ours_backbone.json"),
        "all": eval_json_acc(f"{q}/clinc_t8_bs64_fake/ours_all.json"),
    }
    R[("clinc", "base")] = {
        # base bf16: 따로 안 돌리고 para 기준선 평가(modify_code/.../para_eval/base/clinc_orig.json, forget/retained) 재사용
        "bf16": eval_json_acc(f"{q}/clinc_t8_bs64_fake/base_none.json")
                or eval_json_acc(f"{RES}/para_eval/base/clinc_orig.json"),
        "backbone": eval_json_acc(f"{q}/clinc_t8_bs64_fake/base_backbone.json"),
        "all": None,
    }
    o3c = f"{q}/o3_clinc_t8_ood{o3_ood}"
    R[("clinc", "O3")] = {
        "bf16": o3_clinc_acc(lambda s: f"{O3C}/test_seed0_small_talk_{s}.json"),
        "backbone": o3_clinc_acc(lambda s: f"{o3c}/backbone_{s}.json"),
        "all": o3_clinc_acc(lambda s: f"{o3c}/all_{s}.json"),
    }
    R[("tofu", "RULE")] = {
        "bf16": eval_json_acc(f"{RES}/repro_tofu_lunar/tofu_moe_fixed_layer16_0/"
                              "lunar_llama3.1-8b-instruct_eval_base_tofu_t19_AG20/forget_tofu_t19_AG20.json"),
        "backbone": eval_json_acc(f"{q}/tofu_t19_fake/ours_backbone.json"),
        "all": eval_json_acc(f"{q}/tofu_t19_fake/ours_all.json"),
    }
    R[("tofu", "base")] = {
        "bf16": eval_json_acc(f"{q}/tofu_t19_fake/base_none.json")
                or eval_json_acc(f"{RES}/para_eval/base/tofu_orig.json"),
        "backbone": eval_json_acc(f"{q}/tofu_t19_fake/base_backbone.json"),
        "all": None,
    }
    R[("tofu", "O3")] = {
        "bf16": eval_json_acc(O3T),
        "backbone": eval_json_acc(f"{q}/o3_tofu_stage19/backbone/ptofu.json"),
        "all": eval_json_acc(f"{q}/o3_tofu_stage19/all/ptofu.json"),
    }
    return R


def cell(v, ref):
    if v is None:
        return "-"
    if ref is None:
        return f"{v:.3f}"
    return f"{v:.3f} ({v - ref:+.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--o3_ood", default="recent")
    a = ap.parse_args()
    R = rows(a.o3_ood)
    L = ["# Quantization attack (NF4 4bit, seed 0) — CLINC t8 / TOFU t19\n",
         "열: bf16 = 원래 성능, backbone = 백본만 4bit, all = 백본 + (RULE: MoE lora/gate, O3: LoRA) 4bit. "
         "괄호 = bf16 대비 Δ. forget 은 높아질수록 지식이 되살아난 것.",
         f"O3 CLINC 는 exact_acc (OOD detector: ood_checkpoints_clincqa_{a.o3_ood}), 나머지는 choice_accuracy. "
         "O3 TOFU 평가는 forget 40 / retained 200 문항 (RULE 은 800 / 3200).\n"]
    for ds in ["clinc", "tofu"]:
        L.append(f"\n## {ds.upper()} {'t8' if ds == 'clinc' else 't19'}\n")
        L.append("| method | split | " + " | ".join(MODES) + " |")
        L.append("|---|---|" + "---|" * len(MODES))
        for meth in ["RULE", "O3", "base"]:
            r = R[(ds, meth)]
            for sp in SPLITS[ds]:
                ref = (r["bf16"] or {}).get(sp)
                cells = [cell((r[m] or {}).get(sp), None if m == "bf16" else ref) for m in MODES]
                L.append(f"| {meth} | {sp} | " + " | ".join(cells) + " |")
    out = os.path.join(HERE, "results", "quant_table.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
