"""readme_main_results 빌드 공용 함수: 원본 복사, markdown 표 파싱, CSV 저장, PNG 표 렌더링."""
import csv
import json
import os
import re
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
from matplotlib import font_manager                              # noqa: E402

REPO = "/home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment"
MC = f"{REPO}/modify_code"
RES = f"{MC}/run_results/eval_basemoe_error/llama3.1-8b-instruct"
OUT = f"{MC}/readme_main_results"

for _f in ("/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
           "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"):
    if os.path.isfile(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False


def load(p):
    return json.load(open(p)) if p and os.path.isfile(p) else None


def copy(src, dst, ignore=()):
    """파일/폴더 복사 (이미 있으면 덮어씀). ignore = glob 패턴."""
    if not os.path.exists(src):
        print("  [missing]", src)
        return
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(*ignore) if ignore else None)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def f3(x):
    return "-" if x is None else f"{x:.3f}"


def d3(x, ref):
    if x is None:
        return "-"
    return f"{x:.3f}" if ref is None else f"{x:.3f} ({x - ref:+.3f})"


def slug(s):
    s = re.sub(r"`[^`]*`", "", s)
    s = re.sub(r"[^\w가-힣]+", "_", s).strip("_")
    return s[:80] or "table"


def parse_md_tables(path):
    """markdown 파일의 모든 표 → [(heading 경로, header, rows)]"""
    heads, out, cur = {}, [], None
    for line in open(path).read().splitlines() + [""]:
        m = re.match(r"^(#{1,4})\s+(.*)", line)
        if m:
            lv = len(m.group(1))
            heads = {k: v for k, v in heads.items() if k < lv}
            heads[lv] = m.group(2).strip()
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cur is None:
                cur = {"head": " / ".join(heads[k] for k in sorted(heads) if k > 1) or
                       heads.get(1, ""), "header": cells, "rows": []}
            elif all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
                continue
            else:
                cur["rows"].append(cells)
        elif cur is not None:
            out.append(cur)
            cur = None
    return out


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def md_table(header, rows):
    L = ["| " + " | ".join(map(str, header)) + " |", "|" + "---|" * len(header)]
    L += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(L)


def md_to_csvs(md_path, out_dir, prefix):
    """md 의 표를 전부 CSV 로. 반환: [(csv 경로, table dict)]"""
    res, seen = [], {}
    for t in parse_md_tables(md_path):
        name = slug(t["head"])
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name += f"_{seen[name]}"
        p = os.path.join(out_dir, f"{prefix}__{name}.csv")
        n = max([len(t["header"])] + [len(r) for r in t["rows"]])
        m = re.fullmatch(r"e(\d+)", t["header"][-1])
        while len(t["header"]) < n:          # 원본 md 헤더가 열 수보다 짧은 경우 (expert 열 e0.. 이어서) 보정
            k = len(t["header"])
            t["header"].append(f"e{k - 1}" if m else f"col{k}")
        write_csv(p, t["header"], t["rows"])
        res.append((p, t))
    return res


def render_table(path, header, rows, title=None, note=None, highlight_rows=None,
                 col_max=46, font=9):
    """표 → PNG. highlight_rows = 강조할 row index 집합, 첫 열이 바뀌면 구분선."""
    rows = [[str(c).replace("**", "").replace("\\", "/").replace("−", "-") for c in r] for r in rows]
    header = [str(c).replace("**", "").replace("\\", "/").replace("−", "-") for c in header]

    def wrap(s):
        if len(s) <= col_max:
            return s
        parts, cur = [], ""
        for w in re.split(r"(\s+)", s):
            if len(cur) + len(w) > col_max and cur:
                parts.append(cur.rstrip())
                cur = w.lstrip()
            else:
                cur += w
        parts.append(cur)
        return "\n".join(parts)

    header = [wrap(h) for h in header]
    rows = [[wrap(c) for c in r] for r in rows]

    def vis_len(s):
        return max(sum(1.7 if ord(ch) > 0x3000 else 1.0 for ch in line) for line in s.split("\n"))

    widths = [max([vis_len(header[j])] + [vis_len(r[j]) for r in rows if j < len(r)]) + 2
              for j in range(len(header))]
    tot = sum(widths)
    nlines = [max(c.count("\n") + 1 for c in header)] + [max(c.count("\n") + 1 for c in r) for r in rows]
    fig_w = max(4.0, tot * 0.075 * font / 9)
    if title:
        fig_w = max(fig_w, sum(1.7 if ord(ch) > 0x3000 else 1.0 for ch in title) * 0.085 * (font + 2) / 9 + 0.3)
    if note:
        import textwrap
        note = note.replace("−", "-")
        note = "\n".join(textwrap.fill(p, width=max(40, int(fig_w * 72 / (max(font - 1.5, 6) * 0.85)))) for p in note.split("\n"))
    row_h = 0.24 * font / 9
    fig_h = sum(nlines) * row_h + (0.45 if title else 0.1) + (0.35 * (note.count("\n") + 1) if note else 0)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=170)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    top = 1 - ((0.42 / fig_h) if title else 0.05 / fig_h)
    bottom = (0.35 * (note.count("\n") + 1) / fig_h) if note else 0.02
    if title:
        fig.text(0.01, 1 - 0.12 / fig_h, title.replace("−", "-"), fontsize=font + 2, fontweight="bold", va="top")
    if note:
        fig.text(0.01, 0.06 / fig_h + bottom * 0.15, note, fontsize=font - 1.5, color="#555", va="bottom")
    tb = ax.table(cellText=rows, colLabels=header, colWidths=[w / tot for w in widths],
                  cellLoc="center", loc="upper center", bbox=[0, bottom, 1, top - bottom])
    tb.auto_set_font_size(False)
    tb.set_fontsize(font)
    for (i, j), cell in tb.get_celld().items():
        cell.set_height(nlines[i] * 0.1)
    hl = set(highlight_rows or [])
    prev = None
    for (i, j), cell in tb.get_celld().items():
        cell.set_edgecolor("#c8c8c8")
        cell.set_linewidth(0.5)
        if i == 0:
            cell.set_facecolor("#2f3e4e")
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor("#fff4d6" if (i - 1) in hl else ("#f6f7f9" if i % 2 == 0 else "white"))
    # 첫 열 그룹 구분
    for i, r in enumerate(rows, start=1):
        if prev is not None and r[0] not in ("", prev):
            for j in range(len(header)):
                tb[i, j].visible_edges = "closed"
                tb[i, j].set_linewidth(0.5)
        prev = r[0] if r[0] else prev
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1, ensure_ascii=False)
