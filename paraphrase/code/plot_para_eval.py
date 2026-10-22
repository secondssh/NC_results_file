"""paraphrase 평가 step별 추이 (seed0): 행 = TOFU / CLINC, 열 = forget / retained / utility1 / utility2.

선 = 원본(검정 굵은 선) + paraphrase 유형 5개. 지표 = choice_accuracy (결과 json 기본 채점).
para eval 은 forget/retained 만 평가 (utility 데이터는 paraphrase 대상이 아님) → utility 칸은 원본만.
python repro_check/plot_para_eval.py   → repro_check/para_eval_steps.png
"""
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "run_results/eval_basemoe_error/llama3.1-8b-instruct")
ORIG = {"tofu": RES + "/repro_tofu_lunar/tofu_moe_fixed_layer16_0",
        "clinc": RES + "/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0"}
SPLITS = {"tofu": ["forget", "retained", "real_authors", "world_facts"],
          "clinc": ["forget", "retained", "mrpc", "rte"]}
# 유형 순서 고정 = 범주 색 슬롯 1..5 (기본 검증 팔레트), 원본 = 중립 잉크
TYPES = [("para", "para (semantic)", "#2a78d6", "o"),
         ("para_ctx", "para_ctx (context)", "#eb6834", "s"),
         ("para_struct", "para_struct (structure)", "#1baf7a", "^"),
         ("para_bt", "para_bt (back-transl.)", "#eda100", "D"),
         ("para_noise", "para_noise (typo)", "#e87ba4", "v")]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load_run(d):
    out = {}
    for f in glob.glob(os.path.join(d, "lunar_*_t*_*", "forget_*.json")):
        t = int(re.search(r"_t(\d+)_", os.path.basename(os.path.dirname(f))).group(1))
        r = json.load(open(f))
        out[t] = {sp: r[sp]["results"]["choice_accuracy"] for sp in r if "results" in r[sp]}
    return out


def series(run, sp):
    ts = sorted(t for t in run if sp in run[t])
    return ts, [run[t][sp] for t in ts]


def main():
    for name in ("NanumGothic", "NanumSquare_ac"):
        if any(name in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = name
            break
    plt.rcParams.update({"axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.titlecolor": INK, "font.size": 10})

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), facecolor=SURFACE)
    for row, ds in enumerate(["tofu", "clinc"]):
        orig = load_run(ORIG[ds])
        para = {ty: load_run(f"{RES}/para_eval/{ds}_{ty}/{os.path.basename(ORIG[ds])}") for ty, *_ in TYPES}
        for col, sp in enumerate(SPLITS[ds]):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            ts, ys = series(orig, sp)
            ax.plot(ts, ys, color=INK, lw=3.2, marker="o", ms=6, label="original", zorder=3, alpha=0.85)
            n_para = 0
            for ty, lab, c, mk in TYPES:
                pts, pys = series(para[ty], sp)
                if pts:
                    n_para += 1
                    ax.plot(pts, pys, color=c, lw=2, marker=mk, ms=6, label=lab, zorder=4,
                            markeredgecolor=SURFACE, markeredgewidth=1)
            if n_para == 0:
                ax.text(0.5, 0.06, "paraphrase 평가 없음\n(utility 데이터는 변형 대상 아님)",
                        transform=ax.transAxes, ha="center", va="bottom", color=MUTED, fontsize=9)
            last = max(orig)
            ax.set_xticks(range(0, last + 1, 1 if last <= 9 else 2))
            ax.set_xlim(-0.5, last + 0.5)
            ax.set_ylim(-0.02, 1.02)
            ax.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.set_title(f"{ds.upper()} — {sp}", fontsize=12, fontweight="bold", loc="left")
            ax.set_xlabel("unlearning step t")
            if col == 0:
                ax.set_ylabel("choice accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.995), labelcolor=INK)
    fig.text(0.5, 0.005, "seed 0. forget: 낮을수록 잘 잊음 / retained·utility: 높을수록 좋음. "
             "para eval 진행 중 — 끝난 step 까지만 표시.", ha="center", color=MUTED, fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    out = os.path.join(ROOT, "repro_check/para_eval_steps.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
