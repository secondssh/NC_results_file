"""Router prompt 마지막 토큰 (생성 시점 실제 routing) → % 표 + step 별 색 confusion matrix.

입력: repro_check/router_metrics/token_cm_csv/{ds}_s{0,1,2}_t{t}_{domain}_last.csv (seed 합산)
출력: readme_main_results/router_acc/last_png/
  {ds}_last_acc_pct.{csv,png}          step 별 F/R/mixed acc %, F 오답 분해 %
  {ds}_binary_cm_all_steps.png         모든 step 이진 CM (row %) 한 장
  {ds}_t{t}_{domain}_cm.png            step 별: 이진 CM + 전체 expert CM (row %, 괄호 = 샘플 수)
"""
import csv
import glob
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from common import MC, OUT, render_table, write_csv

SRC = f"{MC}/repro_check/router_metrics/token_cm_csv"
D = f"{OUT}/router_acc/last_png"
# sequential = 한 hue (blue) light→dark, 0 은 surface 쪽으로
CMAP = LinearSegmentedColormap.from_list("blue_seq", ["#ffffff", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
                                                      "#256abf", "#184f95", "#0d366b"])
CMAP.set_bad("#e6e6e6")
INK, MUTED = "#1f2328", "#6b7280"
CLINC_DOMS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining", "auto_and_commute",
              "utility", "small_talk"]


def load_cm(ds):
    """→ {t: (domain, rows, cols, summed matrix)}"""
    out = {}
    for f in sorted(glob.glob(f"{SRC}/{ds}_s*_t*_last.csv")):
        m = re.match(rf"{ds}_s(\d)_t(\d+)_(.+)_last\.csv", os.path.basename(f))
        t, dom = int(m.group(2)), m.group(3)
        r = list(csv.reader(open(f)))
        cols, rows = r[0][1:], [x[0] for x in r[1:]]
        mat = np.array([[int(v) for v in x[1:]] for x in r[1:]])
        if t in out:
            out[t] = (dom, rows, cols, out[t][3] + mat)
        else:
            out[t] = (dom, rows, cols, mat)
    return dict(sorted(out.items()))


def binary(mat):
    """row0 = R(e0), row i = F_i (정답 expert = e_i) → (F 맞는 e, F 다른 forget e, F e0, R e0, R forget e)"""
    R = mat[0]
    F = mat[1:]
    f_ok = sum(F[i, i + 1] for i in range(len(F)))
    f_e0 = F[:, 0].sum()
    f_other = F.sum() - f_ok - f_e0
    return f_ok, f_other, f_e0, R[0], R.sum() - R[0]


def heat(ax, pct, cnt, xlabels, ylabels, fs, title=None, mask=None):
    data = np.ma.masked_array(pct, mask=mask if mask is not None else np.zeros_like(pct, bool))
    ax.imshow(data, cmap=CMAP, vmin=0, vmax=100, aspect="auto")
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            if mask is not None and mask[i, j]:
                ax.text(j, i, "–", ha="center", va="center", fontsize=fs, color=MUTED)
                continue
            v = pct[i, j]
            if (cnt is not None and cnt[i, j] == 0) or (cnt is None and v == 0):
                txt = "0"
            else:
                txt = f"{v:.1f}%" + (f"\n({cnt[i, j]})" if cnt is not None else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs, color="white" if v > 55 else INK)
    ax.set_xticks(range(len(xlabels)), xlabels, fontsize=max(fs + 1, 8.5))
    ax.set_yticks(range(len(ylabels)), ylabels, fontsize=max(fs + 1, 8.5))
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-.5, len(xlabels)), minor=True)
    ax.set_yticks(np.arange(-.5, len(ylabels)), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=max(fs + 2, 10), loc="left", color=INK)


BIN_X = ["정답 forget expert", "다른 forget expert", "e0 (retain 경로)"]


def bin_mats(b):
    f_ok, f_other, f_e0, r_e0, r_f = b
    nf, nr = f_ok + f_other + f_e0, r_e0 + r_f
    cnt = np.array([[f_ok, f_other, f_e0], [0, r_f, r_e0]])
    pct = np.array([[f_ok / nf, f_other / nf, f_e0 / nf], [0, r_f / nr, r_e0 / nr]]) * 100
    mask = np.array([[False] * 3, [True, False, False]])
    return pct, cnt, mask, nf, nr


def main():
    os.makedirs(D, exist_ok=True)
    for ds in ("clinc", "tofu"):
        cms = load_cm(ds)
        header = ["step", "F acc % (정답 expert)", "F→다른 forget e %", "F→e0 %", "R acc % (e0)", "R→forget e %",
                  "mixed acc %", "F→아무 forget e % (이진)", "n F", "n R"]
        rows, accs = [], []
        for t, (dom, _, _, mat) in cms.items():
            b = binary(mat)
            pct, cnt, mask, nf, nr = bin_mats(b)
            mixed = (b[0] + b[3]) / (nf + nr) * 100
            rows.append([f"t{t}_{dom}", f"{pct[0, 0]:.1f}", f"{pct[0, 1]:.1f}", f"{pct[0, 2]:.1f}", f"{pct[1, 2]:.1f}",
                         f"{pct[1, 1]:.1f}", f"{mixed:.1f}", f"{pct[0, 0] + pct[0, 1]:.1f}", nf, nr])
            accs.append((pct[0, 0], pct[1, 2], mixed, pct[0, 0] + pct[0, 1]))
        a = np.array(accs).mean(0)
        rows.append(["평균 (step)", f"{a[0]:.1f}", "", "", f"{a[1]:.1f}", "", f"{a[2]:.1f}", f"{a[3]:.1f}", "", ""])
        write_csv(f"{D}/{ds}_last_acc_pct.csv", header, rows)
        render_table(f"{D}/{ds}_last_acc_pct.png", header, rows, col_max=14, highlight_rows=[len(rows) - 1],
                     title=f"{ds.upper()} prompt 마지막 토큰 router acc (%) — 생성 시점 실제 routing, seed 0/1/2 합",
                     note="F acc = forget 문항이 그 step 의 정답 expert 로 간 비율, R acc = retained 가 e0 로 간 비율, "
                          "mixed = forget+retained micro. F→아무 forget e = forget/retain 만 구분한 이진 정확도.")

        # 모든 step 이진 CM 한 장
        n = len(cms)
        nc = 5 if n > 9 else 3
        nr_ = int(np.ceil(n / nc))
        fig, axes = plt.subplots(nr_, nc, figsize=(nc * 4.3, nr_ * 2.15), dpi=160)
        for ax, (t, (dom, _, _, mat)) in zip(axes.flat, cms.items()):
            pct, cnt, mask, nf, nr = bin_mats(binary(mat))
            heat(ax, pct, cnt, ["정답 F e", "다른 F e", "e0"], ["F", "R"], 7.5,
                 title=f"t{t} {dom}  F {pct[0, 0]:.1f}% / R {pct[1, 2]:.1f}%", mask=mask)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        fig.suptitle(f"{ds.upper()} 이진 confusion (prompt 마지막 토큰, row %, 괄호 = 샘플 수, seed 합)",
                     fontsize=12, x=0.01, ha="left", color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(f"{D}/{ds}_binary_cm_all_steps.png", facecolor="white")
        plt.close(fig)

        # 한 격자: 행 = step, 열 = [Forget → 정답 e / 다른 forget e / e0] | [Retain → e0 / forget e]
        ylab, P, C = [], [], []
        for t, (dom, _, _, mat) in cms.items():
            pct, cnt, _, _, _ = bin_mats(binary(mat))
            ylab.append(f"t{t} {dom}")
            P.append([pct[0, 0], pct[0, 1], pct[0, 2], pct[1, 2], pct[1, 1]])
            C.append([cnt[0, 0], cnt[0, 1], cnt[0, 2], cnt[1, 2], cnt[1, 1]])
        P, C = np.array(P), np.array(C)
        h = 0.55 * n + 1.6
        fig, (axf, axr) = plt.subplots(1, 2, figsize=(10.5, h), dpi=170, sharey=True,
                                       gridspec_kw={"width_ratios": [3, 2], "wspace": 0.04})
        fs = 8.5 if n <= 10 else 7.5
        heat(axf, P[:, :3], C[:, :3], ["정답 forget e", "다른 forget e", "e0"], ylab, fs,
             title="Forget 문항 → 예측")
        heat(axr, P[:, 3:], C[:, 3:], ["e0", "forget e"], ylab, fs, title="Retain 문항 → 예측")
        axr.tick_params(axis="y", left=False, labelleft=False)
        for ax in (axf, axr):
            ax.xaxis.tick_top()
        fig.suptitle(f"{ds.upper()} router prompt 마지막 토큰 — step 별 이진 confusion (row %, 괄호 = 샘플 수, seed 합)",
                     fontsize=11.5, x=0.01, ha="left", color=INK, fontweight="bold")
        fig.text(0.01, 0.005, "정답 칸 = Forget→정답 forget e (F acc), Retain→e0 (R acc). 각 블록의 행 합 = 100%.",
                 fontsize=8, color=MUTED)
        lw = max(len(y) for y in ylab) * 0.085 + 0.3          # y 라벨 폭 (inch)
        fig.subplots_adjust(left=lw / 10.5, right=0.99, top=1 - 1.05 / h, bottom=0.45 / h, wspace=0.04)
        fig.savefig(f"{D}/{ds}_binary_cm_grid.png", facecolor="white")
        plt.close(fig)

        # step 별: 이진 + 전체 expert CM
        for t, (dom, rnames, cnames, mat) in cms.items():
            pct_b, cnt_b, mask_b, nf, nr = bin_mats(binary(mat))
            k = mat.shape[0]
            rs = mat.sum(1, keepdims=True)
            pct_f = np.where(rs > 0, mat / np.maximum(rs, 1) * 100, 0)
            if ds == "clinc":
                ylab = ["R (e0)"] + [f"F{i} {CLINC_DOMS[i - 1]}" for i in range(1, k)]
            else:
                ylab = ["R (e0)"] + [f"F{i} AG{i:02d}" for i in range(1, k)]
            fs_f = 8 if k <= 10 else (7 if k <= 14 else 6.3)
            side = max(5.0, 0.62 * k + 1.8)
            fig = plt.figure(figsize=(side + 5.4, side * 0.86 + 0.6), dpi=160)
            gs = fig.add_gridspec(2, 2, width_ratios=[4.6, side], height_ratios=[1, 2.2])
            ax1 = fig.add_subplot(gs[0, 0])
            heat(ax1, pct_b, cnt_b, BIN_X, ["Forget", "Retain"], 8.5, mask=mask_b,
                 title=f"이진: F acc {pct_b[0, 0]:.1f}%  R acc {pct_b[1, 2]:.1f}%")
            ax1.tick_params(axis="x", labelrotation=15)
            ax2 = fig.add_subplot(gs[:, 1])
            heat(ax2, pct_f, mat if k <= 12 else None, cnames, ylab, fs_f,
                 title="전체 expert (row = target, col = 예측 expert)")
            ax2.set_xlabel("예측 expert", fontsize=9, color=MUTED)
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.axis("off")
            diag = [pct_f[0, 0]] + [pct_f[i, i] for i in range(1, k)]
            mixed = (np.trace(mat)) / mat.sum() * 100
            ax3.text(0, 0.95, f"샘플: forget {nf}, retained {nr} (seed 0/1/2 합)\n"
                              f"mixed acc {mixed:.1f}%\n"
                              f"expert 별 정답률 (대각): 최저 {min(diag[1:]):.1f}% ({ylab[1 + int(np.argmin(diag[1:]))]})",
                     va="top", fontsize=9, color=INK, transform=ax3.transAxes)
            fig.suptitle(f"{ds.upper()} t{t} {dom} — router prompt 마지막 토큰 confusion (row %)",
                         fontsize=13, x=0.01, ha="left", color=INK, fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            fig.savefig(f"{D}/{ds}_t{t:02d}_{dom}_cm.png", facecolor="white")
            plt.close(fig)
        print(ds, "done", len(cms), "steps")


if __name__ == "__main__":
    main()
