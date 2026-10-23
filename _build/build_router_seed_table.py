"""Router prompt 마지막 토큰 — F acc / F→다른 forget expert / R acc 를 seed 0/1/2 평균 ± 표준편차로.

입력: repro_check/router_metrics/token_cm_csv/{ds}_s{s}_t{t}_{domain}_last.csv (seed 별)
출력: readme_main_results/router_acc/last_png/{ds}_router_acc_seed_mean_std.{csv,png}
"""
import csv
import glob
import os
import re

import numpy as np

from build_router_last_png import D, SRC, binary
from common import render_table, write_csv


def per_seed(ds):
    """→ {t: (domain, {seed: (F acc, F→다른, R acc)})}"""
    out = {}
    for f in sorted(glob.glob(f"{SRC}/{ds}_s*_t*_last.csv")):
        m = re.match(rf"{ds}_s(\d)_t(\d+)_(.+)_last\.csv", os.path.basename(f))
        s, t, dom = int(m.group(1)), int(m.group(2)), m.group(3)
        r = list(csv.reader(open(f)))
        mat = np.array([[int(v) for v in x[1:]] for x in r[1:]])
        f_ok, f_other, f_e0, r_e0, r_f = binary(mat)
        nf, nr = f_ok + f_other + f_e0, r_e0 + r_f
        out.setdefault(t, (dom, {}))[1][s] = (f_ok / nf * 100, f_other / nf * 100, r_e0 / nr * 100)
    return dict(sorted(out.items()))


def ms(v):
    return f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}"


def main():
    for ds in ("tofu", "clinc"):
        P = per_seed(ds)
        header = ["step", "F acc % (정답 expert)", "F→다른 forget e %", "R acc % (e0)"]
        rows, full, allv = [], [], []
        for t, (dom, S) in P.items():
            v = np.array([S[s] for s in sorted(S)])           # seeds × 3
            allv.append(v)
            rows.append([f"t{t}_{dom}"] + [ms(v[:, j]) for j in range(3)])
            full.append([f"t{t}_{dom}"] + [round(x, 2) for s in sorted(S) for x in S[s]])
        A = np.stack(allv)                                       # steps × seeds × 3
        step_mean = A.mean(0)                                    # seed 별 step 평균 → seed 간 평균 ± sd
        rows.append(["평균 (step)"] + [ms(step_mean[:, j]) for j in range(3)])
        write_csv(f"{D}/{ds}_router_acc_seed_mean_std.csv", header, rows)
        write_csv(f"{D}/{ds}_router_acc_per_seed.csv",
                  ["step"] + [f"s{s} {h}" for s in range(3) for h in header[1:]], full)
        render_table(f"{D}/{ds}_router_acc_seed_mean_std.png", header, rows, highlight_rows=[len(rows) - 1], col_max=24,
                     title=f"{ds.upper()} prompt 마지막 토큰 router acc (%) — 생성 시점 실제 routing, seed 0/1/2 평균 ± 표준편차",
                     note="F acc = forget 문항이 그 문항의 정답 forget expert 로 간 비율, "
                          "F→다른 forget e = forget 문항이 정답이 아닌 다른 forget expert 로 간 비율, "
                          "R acc = retained 문항이 e0 로 간 비율. 표준편차 = seed 3개의 표본 표준편차. "
                          "평균 (step) = seed 별 step 평균의 seed 간 평균 ± 표준편차.")
        print(ds, "done")


if __name__ == "__main__":
    main()
