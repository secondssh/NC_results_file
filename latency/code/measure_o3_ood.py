#!/usr/bin/env python3
"""O3 OOD detector (점수 계산) latency + 메모리 — detector 를 저장한 환경 (o3 env, peft 0.10.0) 에서.

계산은 quant_attack/eval_quant_o3_clinc.py (= eval_o3_clinc.py) 평가 루프와 같다 (batch 1):
  RobertaForSelector_inference('roberta-large', stage LoRA) + mean/precision/fea list (GPU)
  → RoBERTa 토큰화 (512 padding) → get_unsup_Mah_score_s → OCSVM score_samples (CPU, sklearn) → obtain_weights → max(0, w)
입력: measure_infer_v2.py / measure_infer_o3.py 와 같은 200 문항, 같은 순서 (CLINC 마지막 시점 forget 100 + retain 100, seed 1234),
  OOD 입력 텍스트 = 원래 평가처럼 instruction (+ input).
측정: warm-up 2 문항 후 200 문항, 문항마다 sync → 계산 → sync. 구간: 토큰화 / RoBERTa+Mahalanobis (GPU) / OCSVM+weight (CPU).
  메모리: detector 올린 뒤 memory_allocated, 계산 중 max_memory_allocated (+ monitor.py 로 nvidia-smi / host RAM).
출력: latency/results/v2/o3_ood_clinc_t{t}.json (문항별 OOD weight 포함 → measure_infer_o3.py 의 LLM 생성에 그대로 사용)
사용 (ScienceQA_expriment 에서): CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 <o3 env python> modify_code/latency/measure_o3_ood.py --steps 0 8
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
REPO = os.path.dirname(MC)
os.chdir(REPO)
sys.path.insert(0, REPO)
from src.ood_model_selector import RobertaForSelector_inference  # noqa: E402
from transformers import RobertaTokenizer  # noqa: E402

CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]
D = os.path.join(REPO, "data_new")
OOD_ROOT = os.path.join(REPO, "ood_ocsvm/ood_checkpoints_clincqa_recent/ood_checkpoints_clincqa_0")


# --- eval_quant_o3_clinc.py 와 같은 식 ---
def gmm_cdf(x, gmm):
    weights = gmm.weights_
    means = gmm.means_.flatten()
    stds = np.sqrt(gmm.covariances_.flatten())
    return np.sum([w * norm.cdf(x, mean, std) for w, mean, std in zip(weights, means, stds)])


def obtain_weights(input_x, gmm, x0):
    cp_x = gmm_cdf(input_x, gmm)
    cp_symmetric_x = gmm_cdf(2 * x0 - input_x, gmm)
    cp_sum = 1 - max(cp_x, cp_symmetric_x) + min(cp_x, cp_symmetric_x)
    cp_sum *= 10
    w_res = math.exp(cp_sum - 2) / (1 + math.exp(cp_sum - 2))
    if w_res > 0.9:
        w_res = 1.2
    elif 0.3 < w_res <= 0.4:
        w_res = w_res
    else:
        w_res = 0
    return w_res
# ---


def pick_items(n):
    j = "_".join(CLINC_DOMAINS)
    F = json.load(open(f"{D}/clincqa_SD_10/clincqa_{j}_test_SD.json"))
    R = json.load(open(f"{D}/clincqa_RD_10/clincqa_not_{j}_test_RD.json"))
    rng = random.Random(1234)
    items = [F[i] for i in sorted(rng.sample(range(len(F)), n))] + [R[i] for i in sorted(rng.sample(range(len(R)), n))]
    rng.shuffle(items)
    return items


def sync():
    torch.cuda.synchronize()


def stats(xs):
    a = np.asarray(xs, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "n": int(a.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results", "v2"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    import peft
    import sklearn
    import transformers
    items = pick_items(a.n)
    texts = [str(x.get("instruction", "")) + (f"\n\n{x['input']}" if x.get("input") else "") for x in items]
    tok = RobertaTokenizer.from_pretrained("roberta-large")
    for t in a.steps:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        p = os.path.join(OOD_ROOT, f"clincqa_{CLINC_DOMAINS[t]}_ood_clincqa")
        m = RobertaForSelector_inference("roberta-large", lora_path=p + "_roberta_ocsvm", projection_dim=100).to("cuda")
        clr = pickle.load(open(p + "_ocsvm.pkl", "rb"))
        gmm = pickle.load(open(p + "_gmm_w_ocsvm.pkl", "rb"))
        x0 = json.load(open(p + "_threshold_ocsvm.json"))[0]
        lists = [torch.load(p + f"_{k}_list_ocsvm.pt", map_location="cuda") for k in ("mean", "precision", "fea")]
        sync()
        weights_MB = torch.cuda.memory_allocated() / 2**20
        n_params = sum(x.numel() for x in m.parameters())

        def score(text):
            sync(); t0 = time.perf_counter()
            x = tok([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            t1 = time.perf_counter()
            mah = m.get_unsup_Mah_score_s(x, *lists)[:, 1:]
            sync(); t2 = time.perf_counter()
            s = clr.score_samples(mah)
            w = max(0, obtain_weights(s, gmm, x0))
            t3 = time.perf_counter()
            return float(w), 1e3 * (t1 - t0), 1e3 * (t2 - t1), 1e3 * (t3 - t2)

        for x in texts[:2]:
            score(x)
        torch.cuda.reset_peak_memory_stats()
        recs = [score(x) for x in texts]
        res = {"t": t, "detector": p, "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                                               "peft": peft.__version__, "sklearn": sklearn.__version__},
               "gpu": torch.cuda.get_device_name(0), "n_items": len(texts), "warmup": 2, "batch_size": 1,
               "detector_params": n_params, "weights_MB": weights_MB,
               "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
               "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20,
               "total_ms": stats([r[1] + r[2] + r[3] for r in recs]),
               "tokenize_ms": stats([r[1] for r in recs]), "roberta_mahalanobis_ms": stats([r[2] for r in recs]),
               "ocsvm_weight_cpu_ms": stats([r[3] for r in recs]),
               "ood_weights": [r[0] for r in recs]}
        out = os.path.join(a.out_dir, f"o3_ood_clinc_t{t}.json")
        json.dump(res, open(out, "w"), indent=1)
        print(f"[o3 ood t{t}] {res['total_ms']['mean']:.1f}±{res['total_ms']['std']:.1f} ms/sample "
              f"(tok {res['tokenize_ms']['mean']:.1f}, RoBERTa+Mah {res['roberta_mahalanobis_ms']['mean']:.1f}, "
              f"OCSVM {res['ocsvm_weight_cpu_ms']['mean']:.1f}), weights {weights_MB:.0f} MB, peak {res['peak_allocated_MB']:.0f} MB, "
              f"mean w {np.mean(res['ood_weights']):.3f} → {out}", flush=True)
        del m, lists; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
