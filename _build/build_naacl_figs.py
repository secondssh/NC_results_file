"""NAACL 본문/부록용 figure (이미 있는 결과만 읽음, GPU 안 씀).

  paraphrase/tables/para_main_2x2.{png,pdf}        행 = CLINC / TOFU, 열 = forget / retain, 원문 + 5유형, y축 확대
  paraphrase/tables/para_main_2x2_delta.{png,pdf}  같은 배치, para − 원문 (gap만)
  relearning_attack/tables/relearn_main.{png,pdf}  공격 epoch(0/1/5) 별 forget acc, RULE vs O3 vs GA/GD/NPO+KL vs base

/home/hai_ssh/.conda/envs/lunar/bin/python modify_code/readme_main_results/_build/build_naacl_figs.py
"""
import csv
import json
import os

import matplotlib.pyplot as plt

from common import OUT

plt.rcParams["font.family"] = "DejaVu Sans"  # 그림 글자는 영어만 → 논문 기본 sans

TYPES = [("para", "Semantic", "#2a78d6", "o"),
         ("para_ctx", "Context", "#eb6834", "s"),
         ("para_struct", "Structure", "#1baf7a", "^"),
         ("para_bt", "Back-transl.", "#eda100", "D"),
         ("para_noise", "Typo", "#e87ba4", "v")]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DS = [("clinc", "CLINC"), ("tofu", "TOFU")]
SPLITS = [("forget", "Forget acc ↓"), ("retained", "Retain acc ↑")]


def style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8, colors=MUTED)


def save(fig, path):
    fig.savefig(path + ".png", dpi=250, bbox_inches="tight", facecolor="white")
    fig.savefig(path + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  saved", path + ".{png,pdf}")


def para_series():
    s = {}
    for r in csv.DictReader(open(f"{OUT}/paraphrase/all_metrics.csv")):
        if r["model"] == "RULE":
            s.setdefault((r["dataset"], r["type"], r["split"]), {})[int(r["step"])] = float(r["choice_accuracy"])
    return {k: [v[t] for t in sorted(v)] for k, v in s.items()}


def para_figs():
    s = para_series()
    for delta in (False, True):
        fig, axes = plt.subplots(2, 2, figsize=(9, 4.6), gridspec_kw=dict(width_ratios=[1, 1]))
        for i, (ds, dname) in enumerate(DS):
            for j, (sp, sname) in enumerate(SPLITS):
                ax = axes[i, j]
                style(ax)
                orig = s[(ds, "orig", sp)]
                x = list(range(len(orig)))
                ys = []
                if delta:
                    ax.axhline(0, color=INK, lw=1.2)
                else:
                    ax.plot(x, orig, color=INK, lw=2.2, marker="o", ms=3, label="Original", zorder=5)
                    ys += orig
                for ty, lab, c, m in TYPES:
                    y = s[(ds, ty, sp)]
                    y = [a - b for a, b in zip(y, orig)] if delta else y
                    ax.plot(x, y, color=c, lw=1.2, marker=m, ms=3, label=lab, alpha=0.9)
                    ys += y
                lo, hi = min(ys), max(ys)
                pad = max(0.15 * (hi - lo), 0.01)
                if delta:
                    m_ = max(abs(lo), abs(hi)) + pad
                    ax.set_ylim(-m_, m_)
                else:
                    ax.set_ylim(max(0, lo - pad), min(1, hi + pad))
                ax.set_xticks(x[::2] if len(x) > 10 else x)
                ax.set_xlim(-0.4, len(x) - 0.6)
                if i == 0:
                    ax.set_title(sname + (" (Para - Orig)" if delta else ""), fontsize=10, color=INK)
                if j == 0:
                    ax.set_ylabel(dname, fontsize=10, color=INK)
                if i == 1:
                    ax.set_xlabel("unlearning step t", fontsize=8, color=MUTED)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.04))
        fig.tight_layout(h_pad=0.6, w_pad=1.2)
        save(fig, f"{OUT}/paraphrase/tables/para_main_2x2" + ("_delta" if delta else ""))


METHODS = [("RULE", "Ours (RULE)", "#d62728", "o", 2.4),
           ("O3", "O3", "#2a78d6", "s", 1.6),
           ("GA", "GA", "#8c564b", "^", 1.4),
           ("GD", "GD", "#1baf7a", "D", 1.4),
           ("NPO+KL", "NPO+KL", "#eda100", "v", 1.4),
           ("base", "Base (before unlearning)", "#7f7f7f", "x", 1.2)]
# RULE / O3 = 백본에 새 LoRA (MoE / O3 LoRA 는 배포된 그대로), GA·GD·NPO+KL = full-FT 모델에 같은 백본 LoRA
ATTACK = {"RULE": "백본 LoRA (MoE 고정)", "O3": "백본 LoRA (O3 LoRA 고정)", "base": "base (unlearning 전)",
          "GA": "백본 LoRA (full-FT)", "GD": "백본 LoRA (full-FT)", "NPO+KL": "백본 LoRA (full-FT)"}
PANELS = [("work_t0", "in", "CLINC t0 (work)\nin-domain relearn (CLINC 100)"),
          ("banking_t3", "in", "CLINC t3 (banking)\nin-domain relearn (CLINC 100)"),
          ("banking_t3", "out", "CLINC t3 (banking)\nout-domain relearn (BANKING77 1k)")]


def relearn_point(d, exp, meth, data, ep):
    if ep == 0:
        return d.get(f"{exp}|{meth}|공격 전", {}).get("forget")
    if exp == "work_t0":
        return d.get(f"{exp}|{meth}|ep{ep}", {}).get("forget")
    return d.get(f"{exp}|{meth}|{ATTACK[meth]}|{data}_ep{ep}", {}).get("forget")


def relearn_fig():
    d = json.load(open(f"{OUT}/relearning_attack/relearn_summary.json"))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (exp, data, title) in zip(axes, PANELS):
        style(ax)
        for meth, lab, c, m, lw in METHODS:
            pts = [(k, relearn_point(d, exp, meth, data, ep)) for k, ep in enumerate((0, 1, 5))]
            pts = [(k, v) for k, v in pts if v is not None]
            if len(pts) < 2:
                continue
            ax.plot(*zip(*pts), color=c, marker=m, lw=lw, ms=5, label=lab, ls="--" if meth == "base" else "-")
        ax.set_xticks([0, 1, 2], ["before", "1 epoch", "5 epochs"])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Forget acc (↑ = knowledge recovered)", fontsize=9, color=INK)
    hs = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            hs.setdefault(l, h)
    order = [lab for _, lab, *_ in METHODS if lab in hs]
    fig.legend([hs[l] for l in order], order,loc="upper center", ncol=6, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    save(fig, f"{OUT}/relearning_attack/tables/relearn_main")


def main():
    para_figs()
    relearn_fig()


if __name__ == "__main__":
    main()
