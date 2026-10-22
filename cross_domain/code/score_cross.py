#!/usr/bin/env python3
"""Cross 입력 평가 채점: 같은 생성문 하나를 F 정답과 R 정답에 각각 기존 채점 함수로 적용.

  CLINC : 출력 label 하나 (src_lunar/utils/clincQA._extract_pred_intent, 원래 exact 지표와 같음)
          F 누출 = label == F intent, R 정답 = label == R intent, 그 외 = 둘 다 아님
  TOFU  : src_lunar/utils/tofu.eval_choice_hit (원래 forget/retain 지표: 생성문 임베딩이 [정답, perturbed 5, 거절문 5]
          중 정답에 가장 가까우면 hit). F 누출 = F 정답 기준 hit, R 정답 = R 정답 기준 hit.
          생성문 전체 임베딩은 두 답이 섞여 희석될 수 있으므로 문장 단위(어느 한 문장이라도 hit) 값도 같이 낸다.
기준선 (같은 문항, 단독 입력): 원래 seed0 그 시점 결과(RULE, 문항 순서 = 그 시점 SD/RD 파일)와
            para_eval/base/{ds}_orig.json(base, 마지막 시점 전체 풀 → 문항 키로 매칭). 저장 json 에는 평균만 있어 다시 채점.
router (RULE): router_cross.py 출력 → prompt 마지막 토큰 expert 가 F/e0/그 외 일 때 각각의 F 누출·R 정답.
            단독 입력 routing 은 repro_check/router_metrics/token/{ds}_s0/*.npz (같은 방법) 에서 같은 문항만 뽑는다.

출력: {res_dir}/cross_table.md, cross_summary.json, samples/{ds}_t{t}_{method}_{order}.jsonl
사용: python multi_attack/score_cross.py --res_dir multi_attack/results --datasets clinc tofu
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODIFY_CODE = os.path.dirname(HERE)
sys.path.insert(0, MODIFY_CODE)

RES = os.path.join(MODIFY_CODE, "run_results/eval_basemoe_error/llama3.1-8b-instruct")
CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]
BASE_SINGLE = {"clinc": f"{RES}/para_eval/base/clinc_orig.json", "tofu": f"{RES}/para_eval/base/tofu_orig.json"}
BASE_POOL = {"clinc": "clinc_t8", "tofu": "tofu_t19"}   # base 단독 평가에 쓴 풀 (manifest 에 F/R 원본 경로)


def dom_of(ds, t):
    return CLINC_DOMAINS[t] if ds == "clinc" else f"AG{t + 1:02d}"


def rule_single_path(ds, t):
    run = "repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0" if ds == "clinc" else "repro_tofu_lunar/tofu_moe_fixed_layer16_0"
    g = glob.glob(f"{RES}/{run}/*_t{t}_{dom_of(ds, t)}/forget_*.json")
    return g[0] if g else None


def single_router_path(ds, t):
    return f"{MODIFY_CODE}/repro_check/router_metrics/token/{ds}_s0/t{t}_{dom_of(ds, t)}.npz"


def qkey(d):
    return (norm_ws(d["question"]), norm_ws(d["answer"]), d.get("domain"))


METHOD_NAME = {"base": "base (unlearning 전)", "ours": "RULE"}


def norm_ws(s):
    return re.sub(r"\s", "", str(s)).lower()


def sentences(text):
    t = (text or "").strip()
    seg = [s for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
    return seg or [t]


def side_hits(ds, gens, parts, side):
    """원래 채점 함수로 문항별 hit (CLINC: 출력 label == intent, TOFU: choice hit)."""
    if ds == "clinc":
        from src_lunar.utils.clincQA import _extract_pred_intent
        return [_extract_pred_intent(g) == p["intent"] for g, p in zip(gens, parts)]
    from src_lunar.utils.tofu import eval_choice_hit
    hit = eval_choice_hit(gens, [p["answer"] for p in parts],
                          perturbed_answers_list=[p.get("perturbed_answer") or [] for p in parts],
                          name="forget" if side == "F" else "retained")
    assert len(hit["eval_results"]) == len(parts)
    return [bool(x) for x in hit["eval_results"]]


_RESCORED = {}


def rescore_single(path, pool):
    """저장 결과(path)의 forget/retained 생성문을 원 문항(pool 의 F/R 원본)으로 다시 채점.
    저장 json 에는 평균만 있으므로 평균이 저장 choice_accuracy 와 같은지 확인. → {qkey: hit}"""
    if path in _RESCORED:
        return _RESCORED[path]
    r = json.load(open(path))
    ds = "clinc" if pool["F_src"].find("clincqa") >= 0 else "tofu"
    out = {}
    for side, sp, key in (("F", "forget", "F_src"), ("R", "retained", "R_src")):
        res = r[sp]["results"]
        data = json.load(open(pool[key]))
        gt = res["generated_text"]
        assert len(gt) == len(data) and all(norm_ws(g[2]) == norm_ws(d["answer"]) for g, d in zip(gt, data)), (path, sp)
        h = side_hits(ds, [g[1] for g in gt], data, side)
        if abs(float(np.mean(h)) - float(res["choice_accuracy"])) > 1e-9:
            print(f"[warn] {path} {sp}: 재채점 {np.mean(h):.5f} != 저장 {res['choice_accuracy']:.5f}")
        for d, x in zip(data, h):
            out[qkey(d)] = x
    _RESCORED[path] = out
    return out


def single_lookup(ds, t, method, manifest):
    """같은 F / R 문항의 단독 입력 hit → (dict qkey→hit) 또는 None."""
    if method == "ours":
        p = rule_single_path(ds, t)
        pool = manifest["sets"][f"{ds}_t{t}"]
    else:
        p = BASE_SINGLE[ds]
        pool = manifest["sets"][BASE_POOL[ds]]
    if not p or not os.path.exists(p):
        return None
    return rescore_single(p, pool)


def single_router(ds, t):
    p = single_router_path(ds, t)
    if not os.path.exists(p):
        return None
    z = np.load(p)
    out = {}
    for sp in ("forget", "retained"):
        ends = np.cumsum(z[f"{sp}_prompt_len"]) - 1
        out[sp] = (z[f"{sp}_prompt_sel"][ends].astype(int), z[f"{sp}_target"].astype(int))
    return out


def score_clinc(items, gens):
    from src_lunar.utils.clincQA import _extract_pred_intent
    f = side_hits("clinc", gens, [it["cross"]["F"] for it in items], "F")
    r = side_hits("clinc", gens, [it["cross"]["R"] for it in items], "R")
    return [{"F_leak": a, "R_ok": b, "pred": _extract_pred_intent(g)} for a, b, g in zip(f, r, gens)]


def score_tofu(items, gens):
    from src_lunar.utils.tofu import eval_choice_hit
    rows = [{} for _ in items]
    agg = {}
    for side, key in (("F", "F_leak"), ("R", "R_ok")):
        parts = [it["cross"][side] for it in items]
        hit = eval_choice_hit(gens, [p["answer"] for p in parts],
                              perturbed_answers_list=[p.get("perturbed_answer") or [] for p in parts],
                              name="forget" if side == "F" else "retained")
        assert len(hit["eval_results"]) == len(items)
        agg[f"{side}_refusal_rate"] = hit["refusal_rate"]
        agg[f"{side}_perturbed_rate"] = hit["perturbed_rate"]
        segs = [sentences(g) for g in gens]
        flat_g, flat_gt, flat_pt, owner = [], [], [], []
        for i, (s, p) in enumerate(zip(segs, parts)):
            for x in s:
                flat_g.append(x); flat_gt.append(p["answer"]); flat_pt.append(p.get("perturbed_answer") or []); owner.append(i)
        seg_hit = eval_choice_hit(flat_g, flat_gt, perturbed_answers_list=flat_pt, name="x")["eval_results"]
        any_hit = [False] * len(items)
        for i, h in zip(owner, seg_hit):
            any_hit[i] |= bool(h)
        for i in range(len(items)):
            rows[i][key] = bool(hit["eval_results"][i])
            rows[i][key + "_seg"] = any_hit[i]
            rows[i]["n_sent"] = len(segs[i])
    return rows, agg


def mean(xs):
    xs = list(xs)
    return float(np.mean(xs)) if xs else float("nan")


def fmt(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res_dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--datasets", nargs="+", default=["clinc", "tofu"])
    a = ap.parse_args()
    res_dir = os.path.abspath(a.res_dir)
    os.makedirs(os.path.join(res_dir, "samples"), exist_ok=True)
    manifest = json.load(open(os.path.join(HERE, "data", "manifest.json")))

    md = ["# Cross domain / cross author 입력 평가 (seed0)", "",
          "F 누출률 = 생성문이 F(forget) 정답으로 채점된 비율 (낮을수록 좋음), R 정답률 = R(retain) 정답으로 채점된 비율.",
          "단독 = 같은 문항을 원래 평가(한 입력에 한 문항)에서 본 값. router = RULE layer16 prompt 마지막 토큰 expert "
          "(F = F 문항의 expert, e0 = retain 경로, other = 다른 forget expert).", ""]
    summary = {}
    done_dt = set()
    for ds, t in [(ds, t) for ds in a.datasets for t in sorted(
            int(m.group(1)) for m in {re.match(rf"{ds}_t(\d+)_(?:ours|base)_FR\.json$", os.path.basename(x))
                                      for x in glob.glob(os.path.join(res_dir, f"{ds}_t*_*.json"))} if m)]:
        if (ds, t) in done_dt:
            continue
        done_dt.add((ds, t))
        n_f = t + 1
        desc = (f"F {n_f}개 도메인 ({CLINC_DOMAINS[0]}..{CLINC_DOMAINS[t]}) / R 남은 도메인" if ds == "clinc"
                else f"F AG01–AG{t + 1:02d} / R AG{t + 2:02d}–AG100")
        md += [f"## {ds.upper()} t{t} ({desc})", ""]
        hdr = "| model | 입력 | n | F 누출률 | R 정답률 | 둘 다 아님 |"
        if ds == "tofu":
            hdr += " F 누출(문장) | R 정답(문장) | 거절(F 기준) |"
        hdr += " router last F / e0 / other | gen tok F / e0 / other |"
        md += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
        breakdown = []
        s_router = single_router(ds, t)
        for method in ("base", "ours"):
            single = single_lookup(ds, t, method, manifest)
            single_done = False
            for order in ("FR", "RF"):
                p = os.path.join(res_dir, f"{ds}_t{t}_{method}_{order}.json")
                if not os.path.exists(p):
                    continue
                r = json.load(open(p))["forget"]
                items = json.load(open(r["data_path"]))
                gt = r["results"]["generated_text"]
                assert len(items) == len(gt), (p, len(items), len(gt))
                assert all(norm_ws(it["answer"]) == norm_ws(g[2]) for it, g in zip(items, gt)), f"{p}: 순서 불일치"
                gens = [g[1] for g in gt]
                agg = {}
                if ds == "clinc":
                    rows = score_clinc(items, gens)
                else:
                    rows, agg = score_tofu(items, gens)
                stored = float(r["results"]["choice_accuracy"])
                if abs(mean(x["F_leak"] for x in rows) - stored) > 1e-9:
                    print(f"[warn] {p}: F 누출률 {mean(x['F_leak'] for x in rows):.5f} != 저장 choice_accuracy {stored:.5f}")

                rt = None
                rp = os.path.join(res_dir, f"router_{ds}_t{t}_{order}.json")
                if method == "ours" and os.path.exists(rp):
                    rt = json.load(open(rp))
                    assert rt["n"] == len(rows)
                    for x, s in zip(rows, rt["samples"]):
                        x.update(router_last=s["last"], router_last_cls=s["last_cls"], f_expert=s["f_expert"],
                                 gen_cnt=s["gen_cnt"])

                # 단독 입력 기준선 (같은 F / R 문항) — 첫 순서에서 한 번만
                if single and not single_done:
                    fh = [single[qkey(it["cross"]["F"])] for it in items]
                    rh = [single[qkey(it["cross"]["R"])] for it in items]
                    fi = [it["cross"]["F"]["src_idx"] for it in items]
                    ri = [it["cross"]["R"]["src_idx"] for it in items]
                    line = f"| {METHOD_NAME[method]} | 단독 (F만 / R만) | {len(items)} | {fmt(mean(fh))} | " \
                           f"{fmt(mean(rh))} | – |"
                    if ds == "tofu":
                        line += " – | – | – |"
                    if method == "ours" and s_router:
                        (fl, ft), (rl, _) = s_router["forget"], s_router["retained"]
                        line += f" F만→F {fmt(mean(fl[i] == ft[i] for i in fi))} / R만→e0 {fmt(mean(rl[i] == 0 for i in ri))} | – |"
                    else:
                        line += " – | – |"
                    md.append(line)
                    summary[f"{ds}_t{t}_{method}_single"] = {"F_leak": mean(fh), "R_ok": mean(rh)}
                    single_done = True

                n = len(rows)
                s = {"n": n, "F_leak": mean(x["F_leak"] for x in rows), "R_ok": mean(x["R_ok"] for x in rows),
                     "neither": mean(not x["F_leak"] and not x["R_ok"] for x in rows), **agg}
                line = f"| {METHOD_NAME[method]} | {order[0]}+{order[1]} | {n} | {fmt(s['F_leak'])} | {fmt(s['R_ok'])} | {fmt(s['neither'])} |"
                if ds == "tofu":
                    s["F_leak_seg"] = mean(x["F_leak_seg"] for x in rows)
                    s["R_ok_seg"] = mean(x["R_ok_seg"] for x in rows)
                    line += f" {fmt(s['F_leak_seg'])} | {fmt(s['R_ok_seg'])} | {fmt(agg['F_refusal_rate'])} |"
                if rt:
                    sm = rt["summary"]; s["router"] = sm
                    line += f" {fmt(sm['last_F'])} / {fmt(sm['last_e0'])} / {fmt(sm['last_other'])} |" \
                            f" {fmt(sm['gen_tok_F'])} / {fmt(sm['gen_tok_e0'])} / {fmt(sm['gen_tok_other'])} |"
                    for cls in ("F", "e0", "other"):
                        sub = [x for x in rows if x["router_last_cls"] == cls]
                        breakdown.append(f"| {order[0]}+{order[1]} | {cls} | {len(sub)} | {fmt(mean(x['F_leak'] for x in sub))} | "
                                         f"{fmt(mean(x['R_ok'] for x in sub))} |")
                else:
                    line += " – | – |"
                md.append(line)
                summary[f"{ds}_t{t}_{method}_{order}"] = s
                with open(os.path.join(res_dir, "samples", f"{ds}_t{t}_{method}_{order}.jsonl"), "w") as f:
                    for it, g, x in zip(items, gens, rows):
                        f.write(json.dumps({"pair_id": it["cross"]["pair_id"], "question": it["question"],
                                            "F_domain": it["cross"]["F"]["domain"], "gen": g, **x},
                                           ensure_ascii=False) + "\n")
        if breakdown:
            md += ["", "RULE: prompt 마지막 토큰 expert 별 분해", "",
                   "| 입력 | router last | n | F 누출률 | R 정답률 |", "|---|---|---|---|---|"] + breakdown
        md.append("")
    open(os.path.join(res_dir, "cross_table.md"), "w").write("\n".join(md))
    json.dump(summary, open(os.path.join(res_dir, "cross_summary.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
