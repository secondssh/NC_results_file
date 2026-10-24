"""Latency / memory 실험 → readme_main_results/latency/"""
import glob
import os

from common import MC, OUT, copy, md_to_csvs, render_table

D = f"{OUT}/latency"
L = f"{MC}/latency"

# latency_table.md 의 표 순서 → 파일 이름 (제목은 PNG 위에 표시)
NAMES = [
    ("params", "1. 추가 파라미터 (router / expert / 합계)"),
    ("infer_clinc_bs1_fixed64", "CLINC 추론 — bs1, 64 tok 고정 (주 표)"),
    ("infer_clinc_bs16_fixed64", "CLINC 추론 — bs16 (eval batch), 64 tok 고정"),
    ("infer_clinc_bs1_natural", "CLINC 추론 — bs1, 원래 생성 설정 (출력 길이 효과 포함)"),
    ("infer_clinc_bs16_natural", "CLINC 추론 — bs16, 원래 생성 설정 (논문 평가 조건)"),
    ("infer_tofu_bs1_fixed64", "TOFU 추론 — bs1, 64 tok 고정 (주 표)"),
    ("infer_tofu_bs8_fixed64", "TOFU 추론 — bs8 (eval batch), 64 tok 고정"),
    ("infer_tofu_bs1_natural", "TOFU 추론 — bs1, 원래 생성 설정 (출력 길이 효과 포함)"),
    ("infer_tofu_bs8_natural", "TOFU 추론 — bs8, 원래 생성 설정 (논문 평가 조건)"),
    ("o3_clinc_breakdown", "O3 CLINC 추론 분해 — OOD 점수 계산 + LLM (bs1)"),
    ("o3_tofu_breakdown", "O3 TOFU 추론 분해 — OOD 점수 계산 + LLM (bs1, o3 env)"),
    ("version_sensitivity", "버전 영향 — lunar (torch 2.5.1) vs o3 env (torch 2.1.1), 같은 base Llama"),
    ("microbench_router_expert", "Router / Expert microbenchmark (layer16 MoE 하나, us/call)"),
    ("train_rule", "RULE 학습 1 stage — 단계별 시간 / 메모리"),
    ("train_baseline", "baseline full-FT 학습 (2 GPU ZeRO-3, t0, 1 epoch 실측)"),
    ("train_o3", "O3 학습 1 stage (CLINC t0) — detector + LoRA"),
    ("scale_micro", "expert 수 E 확장 — layer16 microbench + 학습 1 step"),
    ("scale_e2e", "expert 수 E 확장 — 실제 생성 (TOFU, 64 tok 고정)"),
    ("scale_combined", "expert 수 E (요청 수) 확장 통합 — 파라미터 / decode·prefill / bs1 latency·TTFT·peak / 학습 1 epoch·peak"),
]


def main():
    # 원본 결과 (오염된 동시 실행 결과 results/v2/concurrent 는 제외)
    copy(f"{L}/latency_table.md", f"{D}/results/latency_table.md")
    copy(f"{L}/existing_metrics.md", f"{D}/results/existing_metrics.md")
    copy(f"{L}/existing_metrics.json", f"{D}/results/existing_metrics.json")
    copy(f"{L}/results/v2/timing", f"{D}/results/timing")                     # 시간 (monitor --light)
    for f in glob.glob(f"{L}/results/v2/*.json"):                             # 메모리 (PSS) + microbench
        copy(f, f"{D}/results/memory/{os.path.basename(f)}")
    copy(f"{L}/results/v2/baseline", f"{D}/results/memory/baseline")          # baseline host PSS
    copy(f"{L}/results/v2/o3_train", f"{D}/results/o3_train")
    copy(f"{L}/results/baseline", f"{D}/results/baseline_timing_v1")          # baseline 시간 (1 차, GPU 2장 idle)
    for f in sorted(glob.glob(f"{L}/*.py") + glob.glob(f"{L}/*.sh")):
        copy(f, f"{D}/code/{os.path.basename(f)}")

    os.makedirs(f"{D}/tables", exist_ok=True)
    for old in glob.glob(f"{D}/tables/*"):
        os.remove(old)
    tabs = md_to_csvs(f"{L}/latency_table.md", f"{D}/tables", "tmp")
    assert len(tabs) == len(NAMES), (len(tabs), len(NAMES))
    for (p, t), (name, title) in zip(tabs, NAMES):
        os.replace(p, f"{D}/tables/{name}.csv")
        wide = len(t["header"]) > 7
        hl = [i for i, r in enumerate(t["rows"]) if r and r[0].startswith(("rule", "RULE", "o3", "O3"))]
        render_table(f"{D}/tables/{name}.png", t["header"], t["rows"], title=title,
                     font=8 if wide else 9, col_max=22 if wide else 40, highlight_rows=hl)
    print("latency done")


if __name__ == "__main__":
    main()
