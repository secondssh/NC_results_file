#!/usr/bin/env python3
"""latency 실험: 이미 있는 데이터로 뽑을 수 있는 것 (GPU 불필요).

A. 추가 파라미터 수
   RULE : layer16 LUNAR_LoRA_MoE_net ckpt (lora_A [E,r,14336], lora_B [E,4096,r], gate [E,14336]+[E]). pretrained_weight 는 원래 down_proj 복사본이라 제외.
   O3   : 주 LLM LoRA adapter (stage 별, safetensors 헤더) + OOD 탐지기 (roberta-large 공유 backbone + stage 별 roberta LoRA + 저장 통계 파일 용량)
   base : O3_LLAMA31_* model.safetensors 헤더
B. RULE 시간 (로그 + 파일 시각으로 재구성, 다른 작업과 GPU 를 같이 쓰지 않은 run 만)
   CLINC = run_logs/repro_clinc_lunar_bs64.log seed0 (GPU1),  TOFU = run_logs/repro_tofu_lunar_seed12.log seed1 (GPU0)
   step 시작(로그) → ckpt mtime(학습 끝) → 결과 json mtime(평가 끝).
   MoE 학습 = tqdm "Epoch [k/20]" 경과 합 (expert + router 가 한 루프에서 같이 학습됨: train_moe.py mode=joint)
   전처리 = 학습 구간 - MoE 학습 (모델 로드 + direction 추출 + activation 추출)
   평가 = split 별 tqdm "Eval {split}" 경과
C. baseline (GA, GD, NPO, NPO+KL) full-FT 학습 시간: unlearning_baseline_new/slurm_logs 의 HF train_runtime
   (다른 서버, 2 GPU DeepSpeed ZeRO-3 + CPU offload → RULE 과 직접 비교 불가, 참고값)

출력: latency/existing_metrics.md, latency/existing_metrics.json
사용: python latency/extract_existing.py   (modify_code 에서)
"""
import datetime as dt
import glob
import json
import os
import re
import struct

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
REPO = os.path.dirname(MC)
RES = f"{MC}/run_results/eval_basemoe_error/llama3.1-8b-instruct"
CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]


# ---------------------------------------------------------------- A. params
def st_numel(path):
    """safetensors 헤더만 읽어 텐서별 원소 수."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return {k: int(np.prod(v["shape"])) for k, v in h.items() if k != "__metadata__"}


def rule_params(ckpt):
    d = torch.load(ckpt, map_location="cpu", weights_only=False)[16]
    E = d["n_experts"]
    la, lb = d["lora_A"].numel(), d["lora_B"].numel()
    g = d["gate_weight"].numel() + d["gate_bias"].numel()
    e0_norm = float(d["lora_B"][0].float().norm() * d["lora_A"][0].float().norm())
    return {"n_experts": E, "lora_A": la, "lora_B": lb, "expert_total": la + lb, "per_expert": (la + lb) // E,
            "router": g, "total": la + lb + g, "e0_lora_norm_product": e0_norm}


def dir_bytes(p):
    return sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)


def params_section():
    out = {}
    for name, md in (("base_clinc", f"{MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA"),
                     ("base_tofu", f"{MC}/train31Llama/trained_model/O3_LLAMA31_TOFU")):
        shards = glob.glob(f"{md}/*.safetensors")
        if shards:
            out[name] = sum(sum(st_numel(x).values()) for x in shards)
    out["rule_clinc"] = {t: rule_params(f"{MC}/outputs/repro_clinc_lunar_bs64/seed_0/t{t}_{CLINC_DOMAINS[t]}/lora_moe.pt")
                         for t in range(9)}
    out["rule_tofu"] = {t: rule_params(f"{MC}/outputs/tofu_ckpt_repro_lunar/seed_0/t{t}_AG{t + 1:02d}/lora_moe.pt")
                        for t in range(20)}
    # O3 CLINC: stage t 의 LLM LoRA = lora_force_random_<domain0>_force_..._<domain t>_force (누적 이름, 각자 하나의 adapter)
    o3c = {}
    root = f"{REPO}/unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_clinc9_origin"
    for t in range(9):
        name = "lora_force_random_" + "_".join(f"{d}_force" for d in CLINC_DOMAINS[:t + 1])
        p = f"{root}/{name}/adapter_model.safetensors"
        if os.path.exists(p):
            o3c[t] = sum(st_numel(p).values())
    out["o3_clinc_llm_lora_per_stage"] = o3c
    ood_c = f"{REPO}/ood_ocsvm/ood_checkpoints_clincqa_recent/ood_checkpoints_clincqa_0"
    rob_c = sorted(glob.glob(f"{ood_c}/*_roberta_ocsvm"))
    out["o3_clinc_ood"] = {"n_stage_roberta_lora": len(rob_c),
                           "roberta_lora_params_per_stage": [sum(st_numel(glob.glob(f"{r}/*.safetensors")[0]).values()) for r in rob_c],
                           "stats_bytes_total": sum(os.path.getsize(f) for f in glob.glob(f"{ood_c}/*") if os.path.isfile(f)),
                           "dir_bytes_total": dir_bytes(ood_c)}
    o3t = {}
    for t in range(20):
        p = f"{os.path.dirname(REPO)}/PTOFU_experiment/files/results/unlearn_ptofu/llama31_8b_O3/seed_0/stage_{t}/checkpoints/adapter_model.safetensors"
        if os.path.exists(p):
            o3t[t] = sum(st_numel(p).values())
    out["o3_tofu_llm_lora_per_stage"] = o3t
    ood_t = f"{os.path.dirname(REPO)}/PTOFU_experiment/ood_checkpoints_seed_0"
    rob_t = sorted(glob.glob(f"{ood_t}/stage_*_roberta"), key=lambda x: int(re.search(r"stage_(\d+)", x).group(1)))
    out["o3_tofu_ood"] = {"n_stage_roberta_lora": len(rob_t),
                          "roberta_lora_params_per_stage": [sum(st_numel(f"{r}/adapter_model.safetensors").values()) for r in rob_t],
                          "stats_bytes_total": sum(os.path.getsize(f) for f in glob.glob(f"{ood_t}/*") if os.path.isfile(f)),
                          "dir_bytes_total": dir_bytes(ood_t)}
    out["roberta_large_backbone_params_note"] = "roberta-large ≈ 355M (공유 backbone, 로컬에 가중치 없음 — HF 공개 수치)"
    return out


# ---------------------------------------------------------------- B. RULE timing
TS = "%Y-%m-%d %H:%M:%S"


def secs(s):
    p = [int(x) for x in s.split(":")]
    return sum(v * 60 ** i for i, v in enumerate(reversed(p)))


def parse_log(path, seed):
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    seeds = re.split(r"^==== \[", txt, flags=re.M)
    seg = [s for s in seeds if re.match(rf"[^\]]+\] SEED={seed}\b", s)]
    assert seg, (path, seed)
    steps = re.split(r"^---- \[", seg[0], flags=re.M)[1:]
    out = []
    for s in steps:
        m = re.match(r"([\d\- :]+)\] step=(\d+) (\S+)", s)
        start, t, dom = dt.datetime.strptime(m.group(1).strip(), TS), int(m.group(2)), m.group(3)
        ep = {}
        for k, tot, n, N, el in re.findall(r"Epoch \[(\d+)/(\d+)\]: 100%\|[^|\n]*\|\s*(\d+)/(\d+) \[([\d:]+)<", s):
            if n == N:
                ep[int(k)] = max(ep.get(int(k), 0), secs(el))
        ev = {}
        for name, n, N, el in re.findall(r"Eval (\w+): 100%\|[^|\n]*\|\s*(\d+)/(\d+) \[([\d:]+)<", s):
            if n == N:
                ev[name] = {"sec": secs(el), "batches": int(N)}
        out.append({"t": t, "domain": dom, "start": start, "epochs": ep, "eval": ev})
    return out


def rule_timing():
    res = {}
    specs = {
        "clinc": dict(log=f"{MC}/run_logs/repro_clinc_lunar_bs64.log", seed=0,
                      ckpt=lambda t, d: f"{MC}/outputs/repro_clinc_lunar_bs64/seed_0/t{t}_{d}/lora_moe.pt",
                      result=lambda t, d: glob.glob(f"{RES}/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0/*_t{t}_{d}/forget_*.json"),
                      n_items=lambda t: {"forget": 450 * (t + 1), "retained": 450 * (9 - t), "mrpc": 1725, "rte": 277}),
        "tofu": dict(log=f"{MC}/run_logs/repro_tofu_lunar_seed12.log", seed=1,
                     ckpt=lambda t, d: f"{MC}/outputs/tofu_ckpt_repro_lunar/seed_1/t{t}_{d}/lora_moe.pt",
                     result=lambda t, d: glob.glob(f"{RES}/repro_tofu_lunar/tofu_moe_fixed_layer16_1/*_t{t}_{d}/forget_*.json"),
                     n_items=lambda t: {"forget": 40 * (t + 1), "retained": 4000 - 40 * (t + 1), "real_authors": 100, "world_facts": 117}),
    }
    for ds, sp in specs.items():
        rows = []
        for s in parse_log(sp["log"], sp["seed"]):
            ck, rj = sp["ckpt"](s["t"], s["domain"]), sp["result"](s["t"], s["domain"])
            if not os.path.exists(ck) or not rj:
                continue
            ck_m = dt.datetime.fromtimestamp(os.path.getmtime(ck))
            rj_m = dt.datetime.fromtimestamp(os.path.getmtime(rj[0]))
            train_wall = (ck_m - s["start"]).total_seconds()
            moe = sum(s["epochs"].values())
            n_items = sp["n_items"](s["t"])
            rows.append({"t": s["t"], "domain": s["domain"], "n_epochs": len(s["epochs"]),
                         "train_wall_s": train_wall, "moe_train_s": moe, "preprocess_s": train_wall - moe,
                         "eval_wall_s": (rj_m - ck_m).total_seconds(),
                         "eval_split_s": {k: v["sec"] for k, v in s["eval"].items()},
                         "eval_s_per_item": {k: v["sec"] / n_items[k] for k, v in s["eval"].items() if k in n_items}})
        res[ds] = rows
    return res


# ---------------------------------------------------------------- C. baselines
def baseline_timing():
    out = {}
    for p in sorted(glob.glob(f"{REPO}/unlearning_baseline_new/slurm_logs/*_ft_*.log")):
        ds = "clinc" if os.path.basename(p).startswith("clinc") else "tofu"
        txt = open(p, errors="ignore").read().replace("\r", "\n")
        cur = None
        for line in txt.split("\n"):
            m = re.search(r"======== method=(\S+) seed=(\d+) step=(\d+)", line)
            if m:
                cur = (m.group(1), int(m.group(3)))
                continue
            m = re.search(r"'train_runtime': ([\d.]+).*'epoch': ([\d.]+)", line)
            if m and cur:
                out.setdefault(ds, {}).setdefault(cur[0], {})[cur[1]] = float(m.group(1))
    return out


# ---------------------------------------------------------------- report
def fmt_n(n):
    return f"{n / 1e6:.3f}M" if n < 1e9 else f"{n / 1e9:.3f}B"


def main():
    P, T, B = params_section(), rule_timing(), baseline_timing()
    md = ["# latency 실험: 기존 데이터에서 뽑은 값", ""]
    md += ["## A. 추가 파라미터 수", "",
           f"base LLM: CLINC {fmt_n(P.get('base_clinc', 0))}, TOFU {fmt_n(P.get('base_tofu', 0))}", "",
           "RULE (layer16 MoE, expert = LoRA r=8 on down_proj, router = Linear(14336→E))", "",
           "| ds | t | experts | expert params (전체 / expert당) | router params | 합계 | base 대비 % |", "|---|---|---|---|---|---|---|"]
    for ds, key in (("CLINC", "rule_clinc"), ("TOFU", "rule_tofu")):
        base = P.get("base_clinc" if ds == "CLINC" else "base_tofu", 8.03e9)
        ts = [0, 4, 8] if ds == "CLINC" else [0, 9, 10, 19]
        for t in ts:
            r = P[key][t]
            md.append(f"| {ds} | t{t} | {r['n_experts']} | {fmt_n(r['expert_total'])} / {fmt_n(r['per_expert'])} | "
                      f"{fmt_n(r['router'])} | {fmt_n(r['total'])} | {100 * r['total'] / base:.4f}% |")
    md += ["", f"*expert 0(retain 경로) LoRA |B||A| = CLINC {P['rule_clinc'][8]['e0_lora_norm_product']:.3g}, "
               f"TOFU {P['rule_tofu'][19]['e0_lora_norm_product']:.3g} (0 이면 e0 는 사실상 원래 down_proj)", ""]
    oc, ot = P["o3_clinc_llm_lora_per_stage"], P["o3_tofu_llm_lora_per_stage"]
    co, to = P["o3_clinc_ood"], P["o3_tofu_ood"]
    md += ["O3 (stage 마다 LLM LoRA 1개 + OOD 탐지기)", "",
           "| ds | LLM LoRA / stage | 저장된 stage adapter 합 (추론 때 몇 개 쓰는지 확인 필요) | OOD roberta LoRA / stage | OOD 통계 파일 합 | OOD 폴더 합 |",
           "|---|---|---|---|---|---|"]
    for ds, o, ood in (("CLINC", oc, co), ("TOFU", ot, to)):
        if o:
            per = list(o.values())
            md.append(f"| {ds} | {fmt_n(per[0])} | {fmt_n(sum(per))} ({len(per)} stage) | "
                      f"{fmt_n(ood['roberta_lora_params_per_stage'][0]) if ood['roberta_lora_params_per_stage'] else '–'} × {ood['n_stage_roberta_lora']} | "
                      f"{ood['stats_bytes_total'] / 1e6:.0f} MB | {ood['dir_bytes_total'] / 1e6:.0f} MB |")
    md += ["", f"*{P['roberta_large_backbone_params_note']}",
           "*O3 추론 (이 repo 의 eval_o3_clinc.py / PTOFU src/model/unlearn.py): LLM 은 마지막 stage adapter 1개('default')만, "
           "OOD 탐지기도 현재 stage 것 1개만 로드 (CLINC: --ood_type 에 현재 도메인 하나, 여러 개 쓰는 코드는 주석 처리 / TOFU: _load_ood_single(task_id)) "
           "→ 추론 추가 파라미터 ≈ 20.97M + 355M (roberta-large) + 1.18M ≈ 377M (8.03B 대비 4.7%), 시점과 무관. 통계(mean/precision/fea list)도 GPU 에 올림", ""]

    md += ["## B. RULE 시간 (step 별, 초)", "",
           "학습 = 전처리(모델 로드 + direction/activation 추출) + MoE 학습(expert+router joint, 20 epoch). 평가 = split 별 tqdm.", ""]
    for ds, rows in T.items():
        splits = sorted({k for r in rows for k in r["eval_split_s"]})
        md += [f"### {ds.upper()} ({'seed0, GPU1' if ds == 'clinc' else 'seed1, GPU0'})", "",
               "| t | 학습 전체 | 전처리 | MoE 학습 (epoch 수) | 평가 전체 | " + " | ".join(f"평가 {s}" for s in splits) + " |",
               "|" + "---|" * (5 + len(splits))]
        for r in rows:
            md.append(f"| t{r['t']} | {r['train_wall_s']:.0f} | {r['preprocess_s']:.0f} | {r['moe_train_s']:.0f} ({r['n_epochs']}) | "
                      f"{r['eval_wall_s']:.0f} | " + " | ".join(f"{r['eval_split_s'].get(s, float('nan')):.0f}" for s in splits) + " |")
        md.append(f"| 평균 | {np.mean([r['train_wall_s'] for r in rows]):.0f} | {np.mean([r['preprocess_s'] for r in rows]):.0f} | "
                  f"{np.mean([r['moe_train_s'] for r in rows]):.0f} | {np.mean([r['eval_wall_s'] for r in rows]):.0f} | " +
                  " | ".join(f"{np.nanmean([r['eval_split_s'].get(s, np.nan) for r in rows]):.0f}" for s in splits) + " |")
        per_item = {s: np.nanmean([r["eval_s_per_item"].get(s, np.nan) for r in rows]) for s in splits}
        md += ["", "평가 문항당 초: " + ", ".join(f"{s} {v:.3f}" for s, v in per_item.items()), ""]

    md += ["## C. baseline full-FT 학습 시간 (HF train_runtime, 초/step) — 다른 서버, 2 GPU ZeRO-3", "",
           "| ds | method | step 수 | 평균 | 중앙값 | 최소–최대 |", "|---|---|---|---|---|---|"]
    for ds, ms in B.items():
        for mth, steps in sorted(ms.items()):
            v = list(steps.values())
            md.append(f"| {ds} | {mth} | {len(v)} | {np.mean(v):.0f} | {np.median(v):.0f} | {min(v):.0f}–{max(v):.0f} |")
    md.append("")
    os.makedirs(HERE, exist_ok=True)
    open(f"{HERE}/existing_metrics.md", "w").write("\n".join(md))
    json.dump({"params": P, "rule_timing": T, "baseline_train_runtime": B},
              open(f"{HERE}/existing_metrics.json", "w"), indent=1, default=str)
    print("\n".join(md))


if __name__ == "__main__":
    main()
