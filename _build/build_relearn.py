"""Relearning attack → readme_main_results/relearning_attack/  (실행 중인 run 이 끝나면 다시 돌리면 갱신됨)"""
import os

from build_quant import BASE_UNITS, DOMS, O3C, O3SP, rule_json
from common import OUT, REPO, RES, copy, d3, f3, load, md_table, render_table, write_csv, write_json

D = f"{OUT}/relearning_attack"
R = f"{REPO}/relearn_attack"
CS = ["forget", "retained", "mrpc", "rte"]


def per_domain(res):
    """eval results → {domain: (hit, n)}  (failure_text = 정답을 낸(=안 잊은) 문항)"""
    fail = {}
    for g in res["failure_text"]:
        fail[g[0]] = fail.get(g[0], 0) + 1
    seen, out = {}, {}
    for g in res["generated_text"]:
        hit = seen.get(g[0], 0) < fail.get(g[0], 0)
        seen[g[0]] = seen.get(g[0], 0) + 1
        h, n = out.get(g[3], (0, 0))
        out[g[3]] = (h + hit, n + 1)
    return out


def ev(p):
    d = load(p)
    if d is None:
        return None
    o = {sp: d[sp]["results"]["choice_accuracy"] for sp in CS if sp in d}
    pd = per_domain(d["forget"]["results"])
    o["per_domain"] = {k: h / n for k, (h, n) in pd.items()}
    bh, bn = pd.get("banking", (0, 0))
    oh = sum(h for k, (h, n) in pd.items() if k != "banking")
    on = sum(n for k, (h, n) in pd.items() if k != "banking")
    o["forget_banking"] = bh / bn if bn else None
    o["forget_other"] = oh / on if on else None
    m = d.get("_attack_meta") or {}
    o["meta"] = {k: m.get(k) for k in ("target", "epochs", "lr", "epoch_loss", "steps", "trainable_params",
                                       "n_lora_modules", "gate_grad_norm_mean", "train_sec", "data")}
    return o


def o3(fn):
    o, meta = {}, {}
    for sp, src in O3SP.items():
        d = load(fn(src))
        if d is not None:
            o[sp] = d["exact_acc"] / 100
            q = d.get("_quant_meta") or {}
            meta.update({k: q[k] for k in ("attack_epochs", "attack_epoch_loss", "attack_train_sec",
                                           "attack_lora_modules", "attack_train_ood_weight_mean") if k in q})
    if not o:
        return None
    o["meta"] = meta
    return o


def main():
    copy(f"{R}/results", f"{D}/results", ignore=("*.pt",))
    copy(f"{R}/data", f"{D}/attack_data")
    for f in ["relearn_clinc.py", "attack_lora.py", "make_data.py", "run_relearn_banking.sh", "run_relearn_t0.sh"]:
        copy(f"{R}/{f}", f"{D}/code/{f}")
    summary, parts = {}, []

    # ---- 1) RULE CLINC t8, 공격 domain banking ----
    pre_rule = ev(rule_json("clinc", 8))
    pre_base = ev(f"{RES}/para_eval/base/clinc_orig.json")
    for sp in ("mrpc", "rte"):                # base mrpc/rte = baseline 의 base 평가 (같은 base 모델)
        pre_base[sp] = BASE_UNITS[sp]["results"]["choice_accuracy"]
    header = ["대상", "공격 데이터", "epoch", "forget(9 domain)", "forget banking", "forget 나머지 8", "retained",
              "mrpc", "rte", "최종 loss", "학습 param"]
    rows, hl = [], []
    for lab, v in [("RULE t8 (공격 전)", pre_rule), ("base (unlearning 전, 공격 전)", pre_base)]:
        summary[lab] = v
        rows.append([lab, "-", "-", f3(v["forget"]), f3(v["forget_banking"]), f3(v["forget_other"]), f3(v["retained"]),
                     f3(v.get("mrpc")), f3(v.get("rte")), "-", "-"])
        hl.append(len(rows) - 1)
    TL = {"backbone": "RULE 백본 LoRA (MoE 고정)", "moe": "RULE 백본 LoRA + MoE", "base": "base (unlearning 전)"}
    DL = {"in": "in: CLINC banking 미사용 100", "out": "out: BANKING77 1000"}
    for t in ("backbone", "moe", "base"):
        ref = pre_base if t == "base" else pre_rule
        for dk in ("in", "out"):
            for e in (1, 5):
                v = ev(f"{R}/results/banking_t8/{t}_{dk}_ep{e}.json")
                if v is None:
                    continue
                summary[f"banking_t8|{t}_{dk}_ep{e}"] = v
                m = v["meta"]
                rows.append([TL[t], DL[dk], e, d3(v["forget"], ref["forget"]),
                             d3(v["forget_banking"], ref["forget_banking"]), d3(v["forget_other"], ref["forget_other"]),
                             d3(v["retained"], ref["retained"]), d3(v.get("mrpc"), ref.get("mrpc")),
                             d3(v.get("rte"), ref.get("rte")),
                             f"{m['epoch_loss'][-1]:.4f}" if m.get("epoch_loss") else "-",
                             f"{m['trainable_params'] / 1e6:.1f}M" if m.get("trainable_params") else "-"])
    write_csv(f"{D}/tables/relearn_banking_t8.csv", header, rows)
    render_table(f"{D}/tables/relearn_banking_t8.png", header, rows, highlight_rows=hl, col_max=24,
                 title="Relearning attack — RULE CLINC t8, 공격 domain = banking (seed0)",
                 note="choice_accuracy. forget 은 높아질수록 되살아난 것. 괄호 = 공격 전 대비 Δ (RULE 행은 RULE t8, base 행은 base 기준).\n"
                      "공격: 새 LoRA r=8 (223 Linear), lr 1e-4, batch 16, 1 epoch(easy) / 5 epoch(hard).")
    parts.append("### RULE CLINC t8 — banking\n\n" + md_table(header, rows))

    # per-domain forget (공격 후)
    doms = DOMS
    h2 = ["조건"] + doms
    r2 = []
    for lab, key in [("RULE t8 공격 전", "RULE t8 (공격 전)"), ("base 공격 전", "base (unlearning 전, 공격 전)")] + \
            [(k.split("|")[1], k) for k in summary if k.startswith("banking_t8|")]:
        v = summary.get(key)
        if v:
            r2.append([lab] + [f3(v["per_domain"].get(d)) for d in doms])
    write_csv(f"{D}/tables/relearn_banking_t8_forget_per_domain.csv", h2, r2)
    render_table(f"{D}/tables/relearn_banking_t8_forget_per_domain.png", h2, r2, highlight_rows=[0, 1], col_max=12,
                 title="Relearning t8 — forget domain 별 choice_accuracy (각 450문항)", font=8)
    parts.append("### t8 forget domain 별\n\n" + md_table(h2, r2))

    # ---- 2) CLINC t0 (work) RULE vs O3 vs GD ----
    W = f"{R}/results/work_t0"
    gd_pre = ev(f"{W}/gd_none.json")
    gd_pre_src = "relearn gd_none (eval bs16)"
    if gd_pre is None:
        gd_pre = ev(f"{REPO}/quant_attack/results/clinc_t0_gd_bs8_fake/ga_gd_none.json")
        gd_pre_src = "quant ga_gd_none (eval bs8, 대체)"
    meths = {
        "RULE": [("공격 전", ev(rule_json("clinc", 0))), ("ep1", ev(f"{W}/rule_ep1.json")), ("ep5", ev(f"{W}/rule_ep5.json"))],
        "O3": [("공격 전", o3(lambda s: f"{O3C}/test_seed0_work_{s}.json")),
               ("ep1", o3(lambda s: f"{W}/o3_ep1_{s}.json")), ("ep5", o3(lambda s: f"{W}/o3_ep5_{s}.json"))],
        "GD": [("공격 전", gd_pre), ("ep1", ev(f"{W}/gd_ep1.json")), ("ep5", ev(f"{W}/gd_ep5.json"))],
    }
    h3 = ["방법", "조건"] + CS + ["공격 loss (epoch 별)", "학습 초"]
    r3, hl3 = [], []
    for meth, lst in meths.items():
        ref = lst[0][1] or {}
        for i, (lab, v) in enumerate(lst):
            summary[f"work_t0|{meth}|{lab}"] = v
            if v is None:
                r3.append([meth if i == 0 else "", lab] + ["(실행 중/미완료)"] + ["-"] * (len(CS) + 1))
                continue
            m = v.get("meta") or {}
            loss = m.get("epoch_loss") or m.get("attack_epoch_loss")
            sec = m.get("train_sec") or m.get("attack_train_sec")
            r3.append([meth if i == 0 else "", lab] +
                      [(f3(v.get(sp)) if i == 0 else d3(v.get(sp), ref.get(sp))) if v.get(sp) is not None else "(미완료)"
                       for sp in CS] +
                      [" / ".join(f"{x:.3g}" for x in loss) if loss and i else "-", sec if sec and i else "-"])
            if i == 0:
                hl3.append(len(r3) - 1)
    write_csv(f"{D}/tables/relearn_work_t0_rule_o3_gd.csv", h3, r3)
    render_table(f"{D}/tables/relearn_work_t0_rule_o3_gd.png", h3, r3, highlight_rows=hl3, col_max=28,
                 title="Relearning attack — CLINC t0 (work), RULE vs O3 vs GD (seed0)",
                 note="공격 데이터 in_work_100 (unlearning 에 안 쓴 work train 문장 100), 백본 새 LoRA r=8, lr 1e-4. 괄호 = 공격 전 대비 Δ.\n"
                      f"RULE/GD = choice_accuracy, O3 = exact_acc (OOD detector recent). GD 공격 전 = {gd_pre_src}.")
    parts.append("### CLINC t0 work — RULE vs O3 vs GD\n\n" + md_table(h3, r3))
    rd = f"{D}/README.md"                   # README 실험 B 표를 csv 와 같은 값으로 갱신
    if os.path.isfile(rd):
        txt = open(rd).read()
        a, b = "<!-- WORK_T0_TABLE -->", "<!-- /WORK_T0_TABLE -->"
        if a in txt and b in txt:
            txt = txt[:txt.index(a) + len(a)] + "\n" + md_table(h3, r3) + "\n" + txt[txt.index(b):]
            open(rd, "w").write(txt)
    write_json(f"{D}/relearn_summary.json", summary)
    open(f"{D}/tables/all_tables.md", "w").write("\n\n".join(parts) + "\n")
    print("relearn done")


if __name__ == "__main__":
    main()
