"""Router prompt 마지막 토큰 — step 별 expert 단위 n×n confusion matrix (정사각형).

행 = 정답 (R → e0, F_i → i 번째 step 에서 지운 domain/author 의 expert e_i), 열 = router 가 실제로 고른 expert.
값 = 행 % (아래 = 문항 수), seed 0/1/2 합. 대각선 = 정답.
출력: readme_main_results/router_acc/last_png/square/{ds}_t{tt}_{domain}_square_cm.png
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from build_router_last_png import CLINC_DOMS, CMAP, D, INK, MUTED, load_cm

OUTD = f"{D}/square"


def main():
    os.makedirs(OUTD, exist_ok=True)
    for ds in ("clinc", "tofu"):
        for t, (dom, _, cols, mat) in load_cm(ds).items():
            k = mat.shape[0]
            pct = mat / np.maximum(mat.sum(1, keepdims=True), 1) * 100
            names = ["R (retain)"] + ([f"F{i} {CLINC_DOMS[i - 1]}" for i in range(1, k)] if ds == "clinc"
                                      else [f"F{i} AG{i:02d}" for i in range(1, k)])
            cell = 0.95 if k <= 10 else (0.62 if k <= 14 else 0.5)
            side = cell * k
            fs = 8 if k <= 10 else (6.5 if k <= 14 else 5.2)
            fig, ax = plt.subplots(figsize=(side + 2.6, side + 1.6), dpi=170)
            ax.imshow(pct, cmap=CMAP, vmin=0, vmax=100)
            for i in range(k):
                for j in range(k):
                    v, c = pct[i, j], mat[i, j]
                    if c == 0:
                        ax.text(j, i, "·", ha="center", va="center", fontsize=fs, color="#b8bec7")
                        continue
                    ax.text(j, i, f"{v:.1f}%\n{c}" if k <= 14 else f"{v:.0f}%\n{c}", ha="center", va="center",
                            fontsize=fs, color="white" if v > 55 else INK, linespacing=1.1)
                ax.add_patch(Rectangle((i - .5, i - .5), 1, 1, fill=False, ec="#c2410c", lw=1.6, zorder=5))
            ax.set_xticks(range(k), cols, fontsize=fs + 1.5)
            ax.set_yticks(range(k), names, fontsize=fs + 1.5)
            ax.xaxis.tick_top()
            ax.set_xlabel("router 가 보낸 expert", fontsize=9, color=MUTED, labelpad=6)
            ax.xaxis.set_label_position("top")
            ax.tick_params(length=0, which="both")
            ax.set_xticks(np.arange(-.5, k), minor=True)
            ax.set_yticks(np.arange(-.5, k), minor=True)
            ax.grid(which="minor", color="white", lw=1.5)
            for s in ax.spines.values():
                s.set_visible(False)
            acc = np.trace(mat) / mat.sum() * 100
            ax.set_title(f"{ds.upper()} t{t} ({dom}) — router prompt 마지막 토큰 {k}×{k} confusion matrix\n"
                         f"행 = 정답 (문항이 속한 곳), 열 = router 가 보낸 expert, 대각선 = 정답 (주황)\n전체 정확도 {acc:.1f}%, 값 = 행 % / 문항 수 (seed 0/1/2 합)",
                         fontsize=10, color=INK, loc="left", pad=30)
            fig.tight_layout()
            fig.savefig(f"{OUTD}/{ds}_t{t:02d}_{dom}_square_cm.png", facecolor="white")
            plt.close(fig)
        print(ds, "done")


if __name__ == "__main__":
    main()
