"""Cross domain / cross author 입력 평가 → readme_main_results/cross_domain/"""
import os

from common import MC, OUT, copy, load, md_to_csvs, md_table, render_table, write_csv

D = f"{OUT}/cross_domain"
M = f"{MC}/multi_attack"


def main():
    copy(f"{M}/results", f"{D}/results")
    copy(f"{M}/data", f"{D}/input_data", ignore=("smoke",))
    for f in ["make_cross_data.py", "run_cross_eval.sh", "router_cross.py", "score_cross.py", "router_span.py",
              "span_summary.py", "run_span.sh", "run_smoke_then_full.sh"]:
        copy(f"{M}/{f}", f"{D}/code/{f}")
    for md, pre in [("cross_table.md", "cross"), ("span_table.md", "span")]:
        for p, t in md_to_csvs(f"{M}/results/{md}", f"{D}/tables", pre):
            name = os.path.basename(p)[:-4]
            title = t["head"] or pre
            if pre == "cross":
                ds, st = t["head"].split()[:2]
                new = (f"analysis_router_lasttoken_{ds.lower()}_{st}" if name.endswith("_2") else f"cross_{ds.lower()}_{st}_main")
                if name.endswith("_2"):
                    title = f"{ds} {st} — RULE: prompt 마지막 토큰 expert 별 F 누출 / R 정답"
            else:
                if "개입" in t["head"]:          # 개입 표는 intervention_table() 이 json 에서 새로 만듦
                    os.remove(p)
                    continue
                new = "analysis_span_routing"
            os.replace(p, f"{D}/tables/{new}.csv")
            name = new
            wide = len(t["header"]) > 8
            hl = [i for i, r in enumerate(t["rows"]) if r and "단독" in " ".join(r)]
            render_table(f"{D}/tables/{name}.png", t["header"], t["rows"], title=title,
                         font=8 if wide else 9, col_max=18 if wide else 34, highlight_rows=hl)
    intervention_table()
    main_table()
    print("cross done")


STEP_DESC = {("clinc", 4): "t4 (F = 5 domain)", ("clinc", 8): "t8 (F = 9 domain)",
             ("tofu", 10): "t10 (F = AG01–AG11)", ("tofu", 19): "t19 (F = AG01–AG20)"}
IN_DESC = {"single": "단독 (F, R 따로 입력)", "FR": "F+R (F 질문이 앞)", "RF": "R+F (R 질문이 앞)"}


def main_table():
    """cross domain 본 결과: 입력만 섞어서 평가 (모델/router 개입 없음)"""
    S = load(f"{M}/results/cross_summary.json")
    header = ["데이터", "시점 (지운 범위)", "모델", "입력", "문항 수", "F 누출률 ↓", "R 정답률 ↑", "둘 다 아님"]
    rows, full, hl = [], [], []
    for (ds, t), sdesc in STEP_DESC.items():
        n = S[f"{ds}_t{t}_ours_FR"]["n"]
        for mi, (m, mlab) in enumerate((("base", "base (unlearning 전)"), ("ours", "RULE"))):
            for ii, inp in enumerate(("single", "FR", "RF")):
                v = S[f"{ds}_t{t}_{m}_{inp}"]
                rows.append([ds.upper() if mi == ii == 0 else "", sdesc if mi == ii == 0 else "",
                             mlab if ii == 0 else "", IN_DESC[inp], n,
                             f"{v['F_leak']:.3f}", f"{v['R_ok']:.3f}", f"{v['neither']:.3f}" if "neither" in v else "-"])
                full.append([ds, t, m, inp, n, v["F_leak"], v["R_ok"], v.get("neither"),
                             v.get("F_leak_seg"), v.get("R_ok_seg"), v.get("F_refusal_rate")])
                if m == "ours" and inp != "single":
                    hl.append(len(rows) - 1)
    write_csv(f"{D}/tables/cross_main_results.csv",
              ["dataset", "step", "model", "input", "n", "F_leak", "R_ok", "neither",
               "F_leak_sentence(TOFU)", "R_ok_sentence(TOFU)", "F_refusal_rate(TOFU)"], full)
    render_table(f"{D}/tables/cross_main_results.png", header, rows, col_max=40, highlight_rows=hl,
                 title="Cross domain / cross author 입력 평가 — forget(F) 질문과 retain(R) 질문을 한 입력에 섞어서 평가 (seed0)",
                 note="모델·router 는 그대로, 입력만 바꿔 원래 평가와 같은 방식으로 생성 평가. 단독 = 같은 문항을 원래처럼 한 문항씩 넣은 결과.\n"
                      "F 누출률 = 생성문이 F 질문의 정답을 맞힌 비율 (낮을수록 잘 잊음), R 정답률 = R 질문의 정답을 맞힌 비율, "
                      "둘 다 아님 = 어느 쪽 정답도 아님. 성공 기준 = F 가 섞여 있으면 F 를 누출하지 않는 것 (R 까지 거절하는 것은 허용).\n"
                      "CLINC 연결: \"{a}, and also {b}\" / TOFU 연결: \"{a} Also, {b}\". 노란 행 = RULE 섞인 입력.")
    open(f"{D}/tables/cross_main_results.md", "w").write(md_table(header, rows) + "\n")


FORCE = {"f2e0": "F 질문 토큰 → e0 (retain 경로)", "f2F": "F 질문 토큰 → 그 F 문항의 forget expert"}


def intervention_table():
    """개입 실험: 한 행 = (데이터, 시점, 입력 순서), 강제 없음 vs 강제 를 나란히"""
    S = f"{M}/results/span"
    header = ["데이터", "시점", "입력 순서", "대상\n문항 수", "강제한 routing (prefill, layer16)",
              "F 누출률\n강제 없음", "F 누출률\n강제", "F 누출률\nΔ",
              "R 정답률\n강제 없음", "R 정답률\n강제", "R 정답률\nΔ",
              "재생성 일치율\n(강제 없음)"]
    rows, full = [], []
    for ds, steps, mode in (("clinc", (4, 8), "f2e0"), ("tofu", (10, 19), "f2F")):
        for t in steps:
            for order, olab in (("FR", "F+R (F 앞)"), ("RF", "R+F (R 앞)")):
                a, b = load(f"{S}/{ds}_t{t}_{order}_int_none.json"), load(f"{S}/{ds}_t{t}_{order}_int_{mode}.json")
                if not (a and b):
                    continue
                rows.append([ds.upper(), f"t{t}", olab, b["n"], FORCE[mode],
                             f"{a['F_leak']:.3f}", f"{b['F_leak']:.3f}", f"{b['F_leak'] - a['F_leak']:+.3f}",
                             f"{a['R_ok']:.3f}", f"{b['R_ok']:.3f}", f"{b['R_ok'] - a['R_ok']:+.3f}",
                             f"{a['gen_eq_saved_rule']:.3f}"])
                full.append(rows[-1] + [f"{a['orig_rule_F_leak']:.3f}", f"{a['orig_rule_R_ok']:.3f}",
                                        f"{b['gen_eq_saved_rule']:.3f}"])
    hc = [h.replace("\n", " ") for h in header]
    write_csv(f"{D}/tables/analysis_routing_intervention.csv",
              hc + ["저장 RULE F 누출률", "저장 RULE R 정답률", "재생성 일치율 (강제)"], full)
    render_table(f"{D}/tables/analysis_routing_intervention.png", header, rows, col_max=40, font=9,
                 title="개입 실험 — prefill 때 F 질문 구간 토큰의 routing 을 강제하고 다시 생성 (RULE, seed0)",
                 note="대상 문항 = 섞인 입력 중 prompt 마지막 토큰이 e0 로 갔고 base(unlearning 전) 는 F 를 답한 문항 "
                      "(TOFU 는 그중 앞 200개).\n"
                      "강제 없음 = 같은 대상 문항을 같은 batch 로 개입 없이 다시 생성한 대조군. Δ = 강제 − 강제 없음. "
                      "F 누출률은 낮을수록, R 정답률은 높을수록 좋음.\n"
                      "CLINC: F 구간을 e0 로 보내면 누출이 늘어나는지 (= F 구간 routing 이 억제에 기여하는지). "
                      "TOFU: F 구간을 forget expert 로 보내면 누출이 줄어드는지.\n"
                      "재생성 일치율 = 강제 없음 재생성문이 원래 저장된 RULE 생성문과 같은 비율 (batch 구성이 달라 100% 아님).")
    open(f"{D}/tables/analysis_routing_intervention.md", "w").write(md_table(hc, rows) + "\n")


if __name__ == "__main__":
    main()
