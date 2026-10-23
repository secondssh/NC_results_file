"""Router prompt 마지막 토큰 — t 시점을 행 묶음으로 넣은 forget/retain confusion matrix 한 장.

행 = (step, 정답 Forget/Retain), 열 = 예측 (정답 forget expert / 오답 forget expert / e0), 값 = 행 %.
Retain 의 정답은 e0 이고 "정답 forget expert" 칸은 해당 없음 (회색).
출력: readme_main_results/router_acc/last_png/{ds}_cm_by_step.png
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from build_router_last_png import CMAP, D, INK, MUTED, bin_mats, binary, load_cm

COLS = ["정답 forget expert", "오답 forget expert", "e0 (retain 경로)"]


def main():
    for ds in ("clinc", "tofu"):
        cms = load_cm(ds)
        P, C, M, ylab, sep = [], [], [], [], []
        for t, (dom, _, _, mat) in cms.items():
            pct, cnt, mask, _, _ = bin_mats(binary(mat))
            for i, tgt in enumerate(("Forget", "Retain")):
                P.append(pct[i]); C.append(cnt[i]); M.append(mask[i])
                ylab.append(f"t{t} {dom}  ·  {tgt}" if i == 0 else f"{tgt}")
            sep.append(len(P))
        P, C, M = np.array(P), np.array(C), np.array(M)
        n = len(P)
        fs = 8.5 if n <= 20 else 7
        fig, ax = plt.subplots(figsize=(8.6, 0.36 * n + 1.5), dpi=170)
        ax.imshow(np.ma.masked_array(P, M), cmap=CMAP, vmin=0, vmax=100, aspect="auto")
        for i in range(n):
            for j in range(3):
                if M[i, j]:
                    ax.text(j, i, "–", ha="center", va="center", fontsize=fs, color=MUTED)
                    continue
                v = P[i, j]
                ax.text(j, i, "0" if C[i, j] == 0 else f"{v:.1f}%  ({C[i, j]})", ha="center", va="center",
                        fontsize=fs, color="white" if v > 55 else INK)
                correct = (i % 2 == 0 and j == 0) or (i % 2 == 1 and j == 2)
                if correct:
                    ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="#c2410c", lw=2.2, zorder=5))
        for k in sep[:-1]:
            ax.axhline(k - .5, color="#1f2328", lw=1.4)
        ax.set_xticks(range(3), COLS, fontsize=9)
        ax.xaxis.tick_top()
        ax.set_yticks(range(n), ylab, fontsize=fs)
        ax.tick_params(length=0, which="both")
        ax.set_xticks(np.arange(-.5, 3), minor=True)
        ax.set_yticks(np.arange(-.5, n), minor=True)
        ax.grid(which="minor", color="white", lw=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.suptitle(f"{ds.upper()} router (prompt 마지막 토큰) — step 별 forget / retain confusion matrix",
                     fontsize=11.5, x=0.01, ha="left", color=INK, fontweight="bold")
        fig.text(0.01, 0.005, "행 = 정답 (Forget / Retain), 열 = router 가 보낸 expert, 값 = 행 % (괄호 = 샘플 수, seed 0/1/2 합). "
                 "주황 테두리 = 정답 칸.", fontsize=7.5, color=MUTED)
        lw = max(len(y) for y in ylab) * 0.075 + 0.3
        fig.subplots_adjust(left=lw / 8.6, right=0.99, top=1 - 0.95 / (0.36 * n + 1.5), bottom=0.35 / (0.36 * n + 1.5))
        fig.savefig(f"{D}/{ds}_cm_by_step.png", facecolor="white")
        plt.close(fig)
        print(ds, "done")


if __name__ == "__main__":
    main()
