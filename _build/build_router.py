"""Router acc → readme_main_results/router_acc/"""
import os

from common import MC, OUT, copy, md_to_csvs, render_table, write_csv

SRC = f"{MC}/repro_check/router_metrics"
D = f"{OUT}/router_acc"
METRICS = [("prompt_last_acc", "last(실제)"), ("saved_router_accuracy", "last(논문 json)"),
           ("prompt_content_tok_acc", "prompt 내용 tok"), ("prompt_tok_acc", "prompt 전체 tok"),
           ("gen_tok_acc", "gen tok"), ("gen_majority_acc", "gen 다수결")]


def main():
    copy(SRC, f"{D}/results")
    for f in ["router_last_token_cm.py", "router_token_eval.py", "summarize_router_token.py",
              "run_router_token_eval.sh", "run_router_token_fill.sh"]:
        copy(f"{MC}/repro_check/{f}", f"{D}/code/{f}")
    for ds, last in [("tofu", "t19"), ("clinc", "t8")]:
        tabs = md_to_csvs(f"{SRC}/token_{ds}.md", f"{D}/tables", f"token_{ds}")
        md_to_csvs(f"{SRC}/last_token_{ds}.md", f"{D}/tables", f"last_token_{ds}")
        T = [t for _, t in tabs]
        summ = next(t for t in T if t["head"].startswith("요약"))
        render_table(f"{D}/tables/{ds}_summary.png", summ["header"], summ["rows"],
                     title=f"{ds.upper()} router acc 요약 (step 평균, seed 0/1/2 평균)",
                     note="target: retained → e0, forget → 해당 step expert. mixed = forget+retain micro. "
                          "last(논문 json) = 결과 json router_accuracy (padding 위치 문제 있음)",
                     highlight_rows=[2, 3])
        # step 별 통합표 (seed mean)
        by = {}
        for t in T:
            for m, _ in METRICS:
                for sp in ("forget", "retained", "mixed"):
                    if t["head"].startswith(sp) and f"`{m}`" in t["head"]:
                        by[(sp, m)] = {r[0]: r[-1] for r in t["rows"]}
        steps = list(by[("forget", "prompt_last_acc")].keys())
        cols = [(sp, m, lab) for sp in ("forget", "retained", "mixed") for m, lab in METRICS
                if not (sp == "mixed" and m in ("prompt_tok_acc", "gen_majority_acc"))]
        header = ["step"] + [f"{sp[0].upper()} {lab}" for sp, _, lab in cols]
        rows = [[s] + [by.get((sp, m), {}).get(s, "-") for sp, m, _ in cols] for s in steps]
        write_csv(f"{D}/tables/{ds}_per_step_seedmean.csv", header, rows)
        render_table(f"{D}/tables/{ds}_per_step_seedmean.png", header, rows,
                     title=f"{ds.upper()} step 별 router acc (seed 평균) — F=forget, R=retained, M=mixed",
                     highlight_rows=[len(rows) - 1], font=8, col_max=14)
        for key, title in [("이진 confusion — prompt last", "forget/retain 이진 confusion — prompt last token (샘플 수, seed 합)"),
                           ("이진 confusion — generated", "forget/retain 이진 confusion — generated token (토큰 수, seed 합)"),
                           (f"전체 confusion matrix {last} seed0 — prompt last", f"전체 CM {last} seed0 — prompt last token (row=target, col=pred)"),
                           (f"전체 confusion matrix {last} seed0 — generated", f"전체 CM {last} seed0 — generated token"),
                           ("Utility split router acc (", "Utility split router acc 요약 (target = e0)")]:
            t = next((t for t in T if key in t["head"]), None)
            if t is None:
                print("  [no table]", ds, key)
                continue
            name = {"이진 confusion — prompt last": "binary_cm_last", "이진 confusion — generated": "binary_cm_gen",
                    "Utility split router acc (": "utility_summary"}.get(key) or \
                ("full_cm_last" if "prompt last" in key else "full_cm_gen")
            render_table(f"{D}/tables/{ds}_{name}.png", t["header"], t["rows"], title=f"{ds.upper()} {title}",
                         font=7 if "full_cm" in name else 9, col_max=12 if "full_cm" in name else 40)
    print("router done")


if __name__ == "__main__":
    main()
