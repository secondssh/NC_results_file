#!/usr/bin/env python3
"""latency / memory 결과 → latency/latency_table.md
  시간 = results/v2/timing/ (monitor --light: nvidia-smi + loadavg 만, 다른 GPU idle)
  host 메모리 = results/v2/ (monitor PSS) — PSS polling 은 큰 프로세스를 느리게 해서 시간은 따로 잼
  GPU 메모리 = torch max_memory_allocated (측정 스크립트 내부, monitor 와 무관) + nvidia-smi peak
  baseline 시간 = results/baseline/ (1차, GPU 2장 idle, nvidia-smi 1초 polling 만), baseline host 메모리 = results/v2/baseline/ (PSS)
  파라미터 = existing_metrics.json"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results", "v2")
T = os.path.join(R, "timing")
B1 = os.path.join(HERE, "results", "baseline")


def J(*p, base=None):
    f = os.path.join(base or R, *p)
    return json.load(open(f)) if os.path.exists(f) else None


def JT(*p):
    return J(*p, base=T)


def pct(a, b):
    return f"{100 * (a / b - 1):+.1f}%"


def main():
    md = ["# Latency / Memory (seed0, A100 80GB PCIe 1장 + 나머지 idle, warm-up 2, 요청마다 sync)", "",
          "시간 = monitor --light 재측정 (results/v2/timing), host 메모리 = PSS 측정 (results/v2), baseline 시간 = 1차 (results/baseline)", ""]
    ex = json.load(open(os.path.join(HERE, "existing_metrics.json")))["params"]
    md += ["## 1. 파라미터", "",
           "| 방법 | router (O3: OOD detector) | expert (O3: unlearning LoRA) | 추론 때 추가 합계 | 학습되는 파라미터 | 8.03B 대비 (추론 추가) |",
           "|---|---|---|---|---|---|"]
    for ds, key, ts in (("CLINC", "rule_clinc", ["0", "4", "8"]), ("TOFU", "rule_tofu", ["0", "10", "19"])):
        for t in ts:
            r = ex[key][t]
            n = r["total"]
            md.append(f"| RULE {ds} t{t} ({r['n_experts']} experts) | {r['router'] / 1e3:.1f}K (Linear 14336→{r['n_experts']}) | "
                      f"{r['expert_total'] / 1e6:.3f}M ({r['per_expert'] / 1e3:.1f}K × {r['n_experts']}) | {n / 1e6:.2f}M | {n / 1e6:.2f}M | {100 * n / 8.03e9:.3f}% |")
    md += ["| O3 (CLINC / TOFU, 시점 무관) | 356.2M (roberta-large 355M 고정 + LoRA 1.18M) + 통계 파일 | 20.97M (LLM LoRA r=8, 7종 linear) | ≈ 377M | "
           "22.15M (LLM LoRA 20.97M + detector LoRA 1.18M) | 4.7% |",
           "| GA / GD / NPO+KL | 0 | 0 | 0 | 8.03B (전체) | 0% |", "",
           "*RULE router = layer16 gate Linear(14336→E) weight + bias = E × 14,337. expert = LoRA r=8 (A 8×14336 + B 4096×8) × E, "
           "e0(retain 경로)의 LoRA 는 0 이지만 텐서로는 저장돼 포함", ""]

    for ds in ("clinc", "tofu"):
        r = JT(f"infer_{ds}.json")
        if not r:
            continue
        pm = J(f"mon_infer_{ds}.json")
        m = r["_meta"]
        keys = [k for k in r if k != "_meta"]
        md += [f"## 2. 추론 latency — {ds.upper()} (lunar, torch {m['torch']}, {m['attn_implementation']}, {m['dtype']}, "
               f"200 문항, 평균 prompt {m['mean_prompt_tokens']:.1f} tok)", ""]
        for setting, label in (("fixed64_bs1", "bs1, 64 tok 고정"), (f"fixed64_evalbs", f"bs{m['eval_bs']}, 64 tok 고정"),
                               ("natural_bs1", "bs1, 원래 생성 설정"), ("natural_evalbs", f"bs{m['eval_bs']}, 원래 생성 설정 (논문 평가)")):
            md += [f"### {label}", "", "| model | ms/sample | vs base | TTFT ms (mean±std, p95) | decode ms/tok | 생성 tok | peak alloc MB |",
                   "|---|---|---|---|---|---|---|"]
            b = r["base"][setting]["ms_per_sample"]["mean"]
            for k in keys:
                x = r[k][setting]
                md.append(f"| {k} | {x['ms_per_sample']['mean']:.1f} ± {x['ms_per_sample']['std']:.1f} | {pct(x['ms_per_sample']['mean'], b)} | "
                          f"{x['ttft_ms']['mean']:.2f} ± {x['ttft_ms']['std']:.2f} ({x['ttft_ms']['p95']:.2f}) | "
                          f"{x['decode_ms_per_token']['mean']:.2f} | {x['mean_new_tokens']:.1f} | {x['mem']['peak_allocated_MB']:.0f} |")
            if setting == "fixed64_bs1":  # O3 = OOD 점수 계산 + LLM 생성 (순차), 같은 문항 / 같은 64 tok
                for t in ((0, 8) if ds == "clinc" else (0, 19)):
                    if ds == "clinc":
                        ood, llm = JT(f"o3_ood_clinc_t{t}.json"), JT(f"o3_llm_clinc_t{t}.json")
                    else:  # TOFU: OOD · LLM 모두 o3 env, 한 파일
                        x = JT(f"o3_tofu_t{t}.json")
                        ood, llm = (x["ood"], x) if x else (None, None)
                    if not (ood and llm):
                        continue
                    f, om = llm["fixed64_bs1"], ood["total_ms"]
                    ms = f["ms_per_sample"]["mean"] + om["mean"]
                    tt = f["ttft_ms"]["mean"] + om["mean"]
                    tts = (f["ttft_ms"]["std"] ** 2 + om["std"] ** 2) ** 0.5
                    md.append(f"| o3_t{t} (OOD 포함) | {ms:.1f} ± {f['ms_per_sample']['std']:.1f} | {pct(ms, b)} | "
                              f"{tt:.2f} ± {tts:.2f} (≤{f['ttft_ms']['p95'] + om['p95']:.2f}) | {f['decode_ms_per_token']['mean']:.2f} | "
                              f"{f['mean_new_tokens']:.1f} | {f['peak_allocated_MB'] + ood['peak_allocated_MB']:.0f} "
                              f"(LLM {f['peak_allocated_MB']:.0f} + detector {ood['peak_allocated_MB']:.0f}) |")
                md += ["", "*O3: ms/sample · TTFT = OOD 점수 계산 + LLM 생성 (순차라 더해짐), peak = LLM + detector. O3 ms/sample std 는 요청별 "
                           "(24 문항 × 3 회), base/RULE/GD 는 3 회 반복 평균 사이. O3 TTFT p95 는 OOD p95 + LLM p95 (상한)"]
            md.append("")
        md += ["가중치 메모리 (MB): " + ", ".join(f"{k} {r[k]['weights_MB']:.0f}" for k in keys), ""]
        if pm:
            md += [f"측정 프로세스 host PSS peak (전 모델 순차 측정 한 번): {pm['host_pss_peak_MB']:.0f} MB, nvidia-smi peak GPU0 {pm['gpu_peak_MiB'].get('0')} MiB", ""]

    o = {t: (JT(f"o3_ood_clinc_t{t}.json"), JT(f"o3_llm_clinc_t{t}.json")) for t in (0, 8)}
    if all(a and b for a, b in o.values()):
        rc = JT("infer_clinc.json")
        md += ["## 3. O3 추론 — CLINC bs1 (OOD = o3 env / detector 저장 환경, LLM = lunar + peft 0.18.1 / LLM LoRA 저장 peft 버전)", "",
               "| | OOD 점수 ms (RoBERTa+Mah / OCSVM CPU) | LLM TTFT ms | TTFT 합 (OOD + LLM) | decode ms/tok | 64 tok 고정 ms/sample (OOD 포함) | "
               "원래 설정 ms/sample (O3 eval max 128, OOD 포함) | 메모리 MB (LLM peak + detector peak) |", "|---|---|---|---|---|---|---|---|"]
        for t, (ood, llm) in o.items():
            f, n = llm["fixed64_bs1"], llm["natural_bs1"]
            md.append(f"| O3 t{t} | {ood['total_ms']['mean']:.1f} ({ood['roberta_mahalanobis_ms']['mean']:.1f} / {ood['ocsvm_weight_cpu_ms']['mean']:.1f}) | "
                      f"{f['ttft_ms']['mean']:.2f} | {f['ttft_incl_ood_ms']:.2f} | {f['decode_ms_per_token']['mean']:.2f} | "
                      f"{f['ms_per_sample_incl_ood']:.1f} | {n['ms_per_sample_incl_ood']:.1f} ({n['mean_new_tokens']:.1f} tok) | "
                      f"{f['peak_allocated_MB']:.0f} + {ood['peak_allocated_MB']:.0f} |")
        if rc:
            for k in ("base", "rule_t0", "rule_t8"):
                x = rc[k]["fixed64_bs1"]
                md.append(f"| {k} (비교) | – | {x['ttft_ms']['mean']:.2f} | {x['ttft_ms']['mean']:.2f} | {x['decode_ms_per_token']['mean']:.2f} | "
                          f"{x['ms_per_sample']['mean']:.1f} | {rc[k]['natural_bs1']['ms_per_sample']['mean']:.1f} "
                          f"({rc[k]['natural_bs1']['mean_new_tokens']:.1f} tok, max 512) | {x['mem']['peak_allocated_MB']:.0f} |")
        md += ["", "*O3 LLM 은 원래 코드의 modeling_llama_hacked_31.py (eager attention) + LoRA 7 종 linear, RULE/base 는 HF 기본 (SDPA)", ""]
    ot3 = {t: JT(f"o3_tofu_t{t}.json") for t in (0, 19)}
    if all(ot3.values()):
        rt = JT("infer_tofu.json")
        md += ["## 3-2. O3 추론 — TOFU bs1 (OOD · LLM 모두 o3 env = 저장 환경, PTOFU 원래 코드)", "",
               "| | OOD 점수 ms (RoBERTa+Mah / OCSVM CPU) | LLM TTFT ms | TTFT 합 (OOD + LLM) | decode ms/tok | 64 tok 고정 ms/sample (OOD 포함) | "
               "원래 설정 ms/sample (O3 eval max 200, OOD 포함) | 메모리 MB (LLM peak + detector peak) |", "|---|---|---|---|---|---|---|---|"]
        for t, x in ot3.items():
            od, f, n = x["ood"], x["fixed64_bs1"], x["natural_bs1"]
            om = od["total_ms"]["mean"]
            md.append(f"| O3 t{t} | {om:.1f} ({od['roberta_mahalanobis_ms']['mean']:.1f} / {od['ocsvm_weight_cpu_ms']['mean']:.1f}) | "
                      f"{f['ttft_ms']['mean']:.2f} | {f['ttft_ms']['mean'] + om:.2f} | {f['decode_ms_per_token']['mean']:.2f} | "
                      f"{f['ms_per_sample']['mean'] + om:.1f} | {n['ms_per_sample']['mean'] + om:.1f} ({n['mean_new_tokens']:.1f} tok) | "
                      f"{f['peak_allocated_MB']:.0f} + {od['peak_allocated_MB']:.0f} |")
        if rt:
            for k in ("base", "rule_t0", "rule_t19", "gd"):
                if k not in rt:
                    continue
                x = rt[k]["fixed64_bs1"]
                md.append(f"| {k} (비교) | – | {x['ttft_ms']['mean']:.2f} | {x['ttft_ms']['mean']:.2f} | {x['decode_ms_per_token']['mean']:.2f} | "
                          f"{x['ms_per_sample']['mean']:.1f} | {rt[k]['natural_bs1']['ms_per_sample']['mean']:.1f} "
                          f"({rt[k]['natural_bs1']['mean_new_tokens']:.1f} tok, max 512) | {x['mem']['peak_allocated_MB']:.0f} |")
        md += ["", "*TOFU O3 OOD 입력은 256 tok (CLINC 512) 이라 detector 가 더 빠름. o3 env 는 lunar 보다 약 6% 빠름 (4절) → O3 에 유리한 쪽", ""]

    v = {e: JT(f"version_{e}.json") for e in ("lunar", "o3")}
    if all(v.values()):
        md += ["## 4. 버전 영향 (같은 base Llama, 같은 prompt 토큰, 64 tok 고정, HF 기본 코드)", "",
               "| attention / bs | lunar (torch 2.5.1, tf 4.46.3) ms/sample · TTFT · decode | o3 env (torch 2.1.1, tf 4.37.2) | 차이 (o3 / lunar) |",
               "|---|---|---|---|"]
        for k in ("sdpa_bs1", "sdpa_bs16", "eager_bs1", "eager_bs16"):
            a, b = v["lunar"][k], v["o3"][k]
            md.append(f"| {k} | {a['ms_per_sample']:.1f} · {a['ttft_ms']:.2f} · {a['decode_ms_per_token']:.2f} | "
                      f"{b['ms_per_sample']:.1f} · {b['ttft_ms']:.2f} · {b['decode_ms_per_token']:.2f} | "
                      f"{pct(b['ms_per_sample'], a['ms_per_sample'])} · {pct(b['ttft_ms'], a['ttft_ms'])} · {pct(b['decode_ms_per_token'], a['decode_ms_per_token'])} |")
        md.append("")

    mb = J("microbench_moe.json")
    if mb:
        md += ["## 5. Router / Expert microbenchmark (layer16 MoE 하나, µs/call, 500 회)", "",
               "| 설정 | experts | 원래 down_proj | router | expert | MoE 전체 | 추가 (MoE - down_proj) |", "|---|---|---|---|---|---|---|"]
        for k, x in mb.items():
            if k == "_meta":
                continue
            md.append(f"| {k} | {x['n_experts']} | {x['base_linear']:.1f} | {x['router']:.1f} | {x['expert']:.1f} | {x['moe_full']:.1f} | {x['added_vs_base']:.1f} |")
        md.append("")

    md += ["## 6. 학습 (1 stage, 시간 = 단계별, MoE 는 1 epoch 실측 × 20)", "",
           "| | 모델 로드 | direction 추출 | activation 추출 | MoE joint (1 epoch × 20) | 합계 (추정) | expert만 / router만 (1 epoch, 참고) | GPU peak MB (torch / nvidia-smi) | host PSS peak MB | activation cache MB |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name in ("train_clinc_t0", "train_clinc_t0_nocopy", "train_clinc_t8_nocopy", "train_tofu_t0_nocopy", "train_tofu_t19_nocopy"):
        d = JT(f"{name}.json")
        mname = f"mon_{name}_copy.json" if name == "train_clinc_t0" else f"mon_{name}.json"
        mon, monl = J(mname), JT(mname)
        if not d:
            continue
        ph = {p["phase"].split(" ")[0]: p for p in d["phases"]}
        load = ph["load_model"]["sec"] + ph["deepcopy"]["sec"]
        dirn = ph["generate_candidate_directions"]["sec"]
        act = ph["prepare_trainset"]["sec"]
        moe = ph["moe_train_joint"]["sec"]
        peak = max(p["peak_allocated_MB"] for p in d["phases"])
        md.append(f"| RULE {d['ds']} t{d['t']} {'(원래 코드, deepcopy)' if not d.get('no_deepcopy') else '(deepcopy 제거)'} | {load:.1f}s | {dirn:.1f}s | {act:.0f}s | "
                  f"{moe:.1f} × {d['full_num_epochs']} = {moe * d['full_num_epochs']:.0f}s | {load + dirn + act + moe * d['full_num_epochs']:.0f}s | "
                  f"{ph['moe_train_expert']['sec']:.1f}s / {ph['moe_train_router']['sec']:.1f}s | "
                  f"{peak:.0f} / {monl['gpu_peak_MiB'].get('0', monl['gpu_peak_MiB'].get(0)) if monl else '–'} | "
                  f"{mon['host_pss_peak_MB']:.0f} | {d['activation_cache']['MB']:.0f} |")
    md.append("")
    sh = {n: JT(f"{n}.json") for n in ("train_clinc_t0", "train_clinc_t0_nocopy")}
    if all(sh.values()):
        same = sh["train_clinc_t0"]["activation_cache"]["sha256"] == sh["train_clinc_t0_nocopy"]["activation_cache"]["sha256"]
        md += [f"*deepcopy 유무로 학습 입력(activation cache) sha256 동일: {same}", ""]
    md += ["### baseline full-FT (2 GPU DeepSpeed ZeRO-3 + optimizer CPU offload, global batch 128, t0, 1 epoch 실측)", "",
           "| | train_runtime 1 epoch | × 10 epoch (원래 설정) | GPU-시간 (× 2 GPU, 10 epoch) | nvidia-smi peak MB (GPU0 / GPU1) | host PSS peak MB |",
           "|---|---|---|---|---|---|"]
    for ds in ("clinc", "tofu"):
        for m in ("ga", "ga_gd", "npo_kl"):
            b = J("baseline", f"{ds}_{m}_t0.json")
            b1 = J(f"{ds}_{m}_t0.json", base=B1)
            if not b or not b1:
                continue
            rt = b1["train_runtime_sec_1epoch"]
            g = b1["nvidia_smi_peak_MiB"]
            md.append(f"| {ds} {m} | {rt:.0f}s | {10 * rt:.0f}s | {20 * rt / 3600:.2f} h | {g.get('0')} / {g.get('1')} | {b['host_pss_peak_MB']:.0f} |")
    md.append("")

    # O3 학습 (CLINC t0, 원래 명령 그대로, GPU 1장)
    ot = {n: J("o3_train", f"mon_{n}.json") for n in ("ood_time", "ood_mem", "lora_time_w", "lora_mem_w")}
    md += ["### O3 학습 (CLINC t0 = work, seed 0, 원래 train_ood.sh / train_unlearn_lora_clinc.sh 인자, GPU 1장)", "",
           "| | 환경 | 시간 (wall, 원래 epoch 전부) | HF train_runtime | nvidia-smi peak MB | host PSS peak MB |", "|---|---|---|---|---|---|"]
    pss = lambda m: f"{m['host_pss_peak_MB']:.0f}" if m else "–"
    if ot["ood_time"]:
        md.append(f"| OOD detector (roberta-large + LoRA, MLM+MoCo 2 epoch + Mahalanobis / OCSVM / GMM) | o3 env (peft 0.10.0) | "
                  f"{ot['ood_time']['wall_sec']:.0f}s | – | {ot['ood_time']['gpu_peak_MiB'].get('0')} | {pss(ot['ood_mem'])} |")
    if ot["lora_time_w"]:
        import re
        lg = os.path.join(HERE, "logs", "v2", "o3_train", "lora_time_w.log")
        rts = re.findall(r"'train_runtime': ([0-9.]+)", open(lg, errors="ignore").read()) if os.path.exists(lg) else []
        md.append(f"| unlearning LoRA (r=8, 7 종 linear, 20 epoch, bs 128) | lunar + peft 0.18.1 (+ chat_template.jinja) | "
                  f"{ot['lora_time_w']['wall_sec']:.0f}s | {float(rts[-1]) if rts else float('nan'):.0f}s | "
                  f"{ot['lora_time_w']['gpu_peak_MiB'].get('0')} | {pss(ot['lora_mem_w'])} (1 epoch 실행) |")
    md += ["", "*O3 1 stage 학습 = detector + LoRA (순차). detector 시간의 대부분은 epoch 마다의 OOD 평가 (retain 13,500 문항 Mahalanobis)", ""]

    sc = JT("scale_experts.json")
    if sc:
        es = sorted(int(k[1:]) for k in sc if k.startswith("E") and k[1:].isdigit())
        md += ["## 7. expert 수 E 에 따른 RULE 비용 (요청 80 건 이상, TOFU)", "",
               "E = 2, 10, 21 실제 ckpt (t0, t8, t19), E ≥ 40 = t19 + 랜덤 expert (연산량은 가중치 값과 무관, 생성은 64 tok 고정만)", "",
               "### layer16 MoE microbench (µs/call, 원래 down_proj 대비 추가) + 학습 1 step", "",
               "| E | 추가 파라미터 | decode bs1 | decode bs8 | prefill bs1 (S=50) | prefill bs8 | prefill bs8 추가 메모리 MB | 학습 1 step ms (batch 64) | 학습 peak MB |",
               "|---|---|---|---|---|---|---|---|---|"]
        for E in es:
            x = sc[f"E{E}"]
            if "micro" not in x:
                continue
            mb = x["micro"]
            md.append(f"| {E}{'' if x['real'] else ' (합성)'} | {x['params'] / 1e6:.2f}M | +{mb['decode_bs1']['added_vs_base']:.0f} | +{mb['decode_bs8']['added_vs_base']:.0f} | "
                      f"+{mb['prefill_bs1']['added_vs_base']:.0f} | +{mb['prefill_bs8']['added_vs_base']:.0f} | {mb['prefill_bs8']['moe_extra_mem_MB']:.0f} | "
                      f"{x['train_step']['ms_per_step']:.1f} | {x['train_step']['peak_allocated_MB']:.0f} |")
        b = sc.get("e2e_base", {}).get("e2e")
        if b:
            md += ["", "### 실제 생성 (TOFU base + layer16 E-expert MoE, 64 tok 고정, 같은 200 문항)", "",
                   "| E | bs1 ms/sample (vs base) | bs1 TTFT | bs1 decode ms/tok | bs8 ms/sample (vs base) | bs8 TTFT | bs8 peak MB (vs E=21) |",
                   "|---|---|---|---|---|---|---|"]
            ref = sc.get("E21", {}).get("e2e")
            for E in ["base"] + [e for e in es if "e2e" in sc[f"E{e}"]]:
                e = b if E == "base" else sc[f"E{E}"]["e2e"]
                f1, f8 = e["fixed64_bs1"], e["fixed64_bs8"]
                dpk = "" if E == "base" or not ref else f" ({f8['mem']['peak_allocated_MB'] - ref['fixed64_bs8']['mem']['peak_allocated_MB']:+.0f})"
                md.append(f"| {E} | {f1['ms_per_sample']['mean']:.1f} ({pct(f1['ms_per_sample']['mean'], b['fixed64_bs1']['ms_per_sample']['mean'])}) | "
                          f"{f1['ttft_ms']['mean']:.2f} | {f1['decode_ms_per_token']['mean']:.2f} | "
                          f"{f8['ms_per_sample']['mean']:.1f} ({pct(f8['ms_per_sample']['mean'], b['fixed64_bs8']['ms_per_sample']['mean'])}) | "
                          f"{f8['ttft_ms']['mean']:.2f} | {f8['mem']['peak_allocated_MB']:.0f}{dpk} |")
            md += ["", "*peak 은 원래 down_proj (112 MB) 를 따로 들고 측정해 base 대비 +112 MB 가 겹쳐 있음 → RULE 의 실제 추가 메모리는 E=21 대비 차이로 볼 것", ""]
        st = JT("scale_train.json")
        if b and st and all("e2e" in sc.get(f"E{E}", {}) for E in es):
            dup = 4096 * 14336 * 2 / 2**20   # 측정 중 따로 들고 있던 원래 down_proj (bf16) = 112 MiB
            bp = b["fixed64_bs1"]["mem"]["peak_allocated_MB"]
            md += ["### 통합 표 — expert 수 E (요청 수) 에 따른 RULE 비용 (TOFU, bs1, 64 tok 고정)", "",
                   "| E (expert) | 추가 파라미터 | decode bs1 추가 (us/token) | prefill bs1 추가 (us) | bs1 ms/sample (vs base) | bs1 TTFT ms | "
                   "bs1 peak MB (RULE 추가) | MoE 학습 1 epoch (실측) | 학습 GPU peak MB |",
                   "|---|---|---|---|---|---|---|---|---|"]
            f1 = b["fixed64_bs1"]
            md.append(f"| base | 0 | – | – | {f1['ms_per_sample']['mean']:.1f} | {f1['ttft_ms']['mean']:.2f} | {bp:.0f} | – | – |")
            for E in es:
                x, e = sc[f"E{E}"], sc[f"E{E}"]["e2e"]["fixed64_bs1"]
                tr = st["per_E"].get(str(E))
                add = e["mem"]["peak_allocated_MB"] - bp - dup
                md.append(f"| {E}{'' if x['real'] else ' (합성)'} | {x['params'] / 1e6:.2f}M | +{x['micro']['decode_bs1']['added_vs_base']:.0f} | "
                          f"+{x['micro']['prefill_bs1']['added_vs_base']:.0f} | {e['ms_per_sample']['mean']:.1f} "
                          f"({pct(e['ms_per_sample']['mean'], f1['ms_per_sample']['mean'])}) | {e['ttft_ms']['mean']:.2f} | "
                          f"{e['mem']['peak_allocated_MB'] - dup:.0f} ({add:+.0f}) | "
                          + (f"{tr['epoch_sec']:.1f}s | {tr['peak_allocated_MB']:.0f} |" if tr else "– | – |"))
            md += ["", f"*decode / prefill 추가 = layer16 MoE microbench (원래 down_proj 대비). bs1 ms/sample · TTFT · peak = 실제 생성 한 회차 "
                       f"(base 와 같은 회차). peak 은 측정 중 따로 들고 있던 원래 down_proj {dup:.0f} MB 를 뺀 값, 괄호 = base 대비 RULE 추가",
                   f"*MoE 학습 1 epoch = TOFU t19 원래 학습 경로로 1 epoch 을 끝까지 돌린 실측 (expert + router joint, 같은 activation, "
                   f"{st['n_train_samples']:,} 샘플 / batch {st['train_batch_size']} = {st['n_train_batches']:,} step, 학습 대상 = 마지막 expert). "
                   f"원래 학습 1 stage = 전처리 (모델 로드 + direction + activation 추출, {st['fixed_sec']['sum_pre_moe']:.0f}s, E 와 무관, 1 회 실측) "
                   f"+ MoE {st['full_num_epochs']} epoch", ""]
    open(os.path.join(HERE, "latency_table.md"), "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
