#!/usr/bin/env python3
"""router_span.py 결과 요약 → {res_dir}/span_table.md

표 1: prompt 구간별 routing (샘플별 비율의 평균). forget = F expert + other forget expert (e0 가 아닌 것)
  그룹: 전체 / 마지막 토큰 e0 & base 가 F 를 답한 문항 (= 개입 대상), 그 안에서 RULE 이 샌 문항 / 막은 문항
표 2: 개입 (F 구간 gate 강제) 전후 F 누출률 / R 정답률. none = 개입 없이 다시 생성 (저장 생성문과 일치율 = 재현 확인)
"""
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def m(xs):
    xs = [x for x in xs if x is not None]
    return f"{np.mean(xs):.3f}" if xs else "–"


def span_cols(rows, key):
    f = [r[key]["F"] for r in rows if r[key]["n"]]
    o = [r[key]["other"] for r in rows if r[key]["n"]]
    e = [r[key]["e0"] for r in rows if r[key]["n"]]
    fo = [a + b for a, b in zip(f, o)]
    return f"{m(f)} / {m(o)} / {m(e)} | {m(fo)}"


def main():
    md = ["# 섞인 입력의 F / R 구간 prompt 토큰 routing (RULE, layer16 top-1)", "",
          "F 구간 → F / other / e0 = F 문항 토큰이 F 문항의 expert / 다른 forget expert / e0(retain 경로) 로 간 비율 (샘플 평균).",
          "forget = F + other. 단독 F = 같은 F 문항을 혼자 넣었을 때의 질문 구간.",
          "대상 = prompt 마지막 토큰이 e0 이고 base 는 F 를 답한 문항 (개입 실험 대상).", "",
          "| ds | t | 입력 | 그룹 | n | F 구간 → F / other / e0 | F 구간 forget | R 구간 forget | 단독 F → F / other / e0 | 단독 F forget |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    files = sorted(f for f in glob.glob(os.path.join(RES, "span", "*.json")) if "_int_" not in f)
    for p in files:
        d = json.load(open(p))
        rows = [r for r in d["rows"] if r]
        tgt = [r for r in rows if r["last_cls"] == "e0" and r["base_F_leak"]]
        groups = [("전체", rows), ("대상", tgt),
                  ("대상 중 RULE 샘", [r for r in tgt if r["rule_F_leak"]]),
                  ("대상 중 RULE 막음", [r for r in tgt if not r["rule_F_leak"]])]
        for name, g in groups:
            if not g:
                continue
            fs, rs, ss = span_cols(g, "F_span"), span_cols(g, "R_span"), span_cols(g, "single_F_span")
            r_forget = rs.split("|")[1].strip()
            md.append(f"| {d['ds']} | t{d['t']} | {d['order'][0]}+{d['order'][1]} | {name} | {len(g)} | "
                      f"{fs.split('|')[0].strip()} | {fs.split('|')[1].strip()} | {r_forget} | "
                      f"{ss.split('|')[0].strip()} | {ss.split('|')[1].strip()} |")

    md += ["", "# 개입: prefill 때 F 구간 토큰의 layer16 gate 강제 → 다시 생성 (대상 문항)", "",
           "f2e0 = F 구간 → e0, f2F = F 구간 → F 문항의 expert, none = 개입 없음 (재현 확인).", "",
           "| ds | t | 입력 | mode | n | F 누출률 | R 정답률 | 원래 RULE F 누출 / R 정답 | 생성문 == 저장 RULE |",
           "|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(glob.glob(os.path.join(RES, "span", "*_int_*.json"))):
        ds, t, order = re.match(r"(\w+?)_t(\d+)_(FR|RF)_int_", os.path.basename(p)).groups()
        r = json.load(open(p))
        md.append(f"| {ds} | t{t} | {order[0]}+{order[1]} | {r['mode']} | {r['n']} | {r['F_leak']:.3f} | {r['R_ok']:.3f} | "
                  f"{r['orig_rule_F_leak']:.3f} / {r['orig_rule_R_ok']:.3f} | {r['gen_eq_saved_rule']:.3f} |")
    open(os.path.join(RES, "span_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
