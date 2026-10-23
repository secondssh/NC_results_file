#!/usr/bin/env python3
"""O3 추론 latency + GPU 메모리 (CLINC). README "### latency 실험" 3).

모델 구성 / OOD 계산은 quant_attack/eval_quant_o3_clinc.py (= eval_o3_clinc.py, quant_mode none) 와 같다:
  LlamaForCausalLM_ood + PeftModel(hacked, 마지막 stage adapter) + OOD 탐지기 1개 (현재 stage 도메인, roberta-large + LoRA,
  mean/precision/fea list GPU) → 입력마다 OOD weight 계산 → model.init_oodweight([1, w]) → generate.
O3 는 OOD weight 가 batch 단위라 원래도 batch 1 → bs1 만 잰다.
입력: measure_infer.py 와 같은 200 문항, 같은 순서 (마지막 시점 forget 100 + retain 100, seed 1234).
  OOD 점수 계산: 200 문항 전부 (ms/sample, tokenize + RoBERTa Mahalanobis + OCSVM + weight)
  LLM: fixed64 / prefill(1 token) 는 앞 24 문항 reps 회, natural(O3 eval 설정 max_new_tokens=128) 은 앞 48 문항 1 회
메모리: LLM+LoRA 올린 뒤 / 탐지기까지 올린 뒤 memory_allocated, 단계별 peak.

실행 (ScienceQA_expriment 에서, peft 0.18.1):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/hai_ssh/pylibs/peft0181 python modify_code/latency/measure_infer_o3.py --steps 0 8
출력: modify_code/latency/results/infer_o3_clinc.json
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
REPO = os.path.dirname(MC)
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(1, os.path.join(REPO, "quant_attack"))
import eval_quant_o3_clinc as q  # noqa: E402  (O3 평가 코드의 import / obtain_weights 재사용)
from transformers import AutoConfig, AutoTokenizer, RobertaTokenizer  # noqa: E402

CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]
D = os.path.join(REPO, "data_new")
BASE = os.path.join(MC, "train31Llama/trained_model/O3_LLAMA31_ClincQA")
LORA_ROOT = os.path.join(REPO, "unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_clinc9_origin")
OOD_ROOT = os.path.join(REPO, "ood_ocsvm/ood_checkpoints_clincqa_recent/ood_checkpoints_clincqa_0")


def sync():
    torch.cuda.synchronize()


def mb(x):
    return x / 2**20


def pick_items(n):
    """measure_infer.py 와 같은 rng 순서로 같은 문항 / 같은 순서."""
    j = "_".join(CLINC_DOMAINS)
    F = json.load(open(f"{D}/clincqa_SD_10/clincqa_{j}_test_SD.json"))
    R = json.load(open(f"{D}/clincqa_RD_10/clincqa_not_{j}_test_RD.json"))
    rng = random.Random(1234)
    items = [F[i] for i in sorted(rng.sample(range(len(F)), n))] + [R[i] for i in sorted(rng.sample(range(len(R)), n))]
    rng.shuffle(items)
    return items


def load_llm(t):
    tok = AutoTokenizer.from_pretrained(BASE, padding_side="left")
    jinja = os.path.join(BASE, "chat_template.jinja")
    if getattr(tok, "chat_template", None) is None and os.path.isfile(jinja):
        tok.chat_template = open(jinja).read()
    if tok.pad_token is None:
        tok.pad_token, tok.pad_token_id = tok.eos_token, tok.eos_token_id
    config = AutoConfig.from_pretrained(BASE)
    config.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    config.orthogonal_loss, config.orthogonal_loss_weight = False, 0.1
    model = q.LlamaForCausalLM_ood.from_pretrained(BASE, config=config, torch_dtype=torch.bfloat16, device_map="auto")
    lora = os.path.join(LORA_ROOT, "lora_force_random_" + "_".join(f"{d}_force" for d in CLINC_DOMAINS[:t + 1]))
    model = q.PeftModel.from_pretrained(model, lora, torch_dtype=torch.bfloat16)
    model.init_olora(orthogonal_loss=False, olora_weights={})
    model.init_active_adapters_d(active_adapters_d=["default"])
    model.config.pad_token_id, model.config.eos_token_id = tok.pad_token_id, tok.eos_token_id
    model.eval()
    return model, tok, lora


def load_ood(t):
    p = os.path.join(OOD_ROOT, f"clincqa_{CLINC_DOMAINS[t]}_ood_clincqa")
    m = q.RobertaForSelector_inference("roberta-large", lora_path=p + "_roberta_ocsvm", projection_dim=100).to("cuda")
    clr = pickle.load(open(p + "_ocsvm.pkl", "rb"))
    gmm = pickle.load(open(p + "_gmm_w_ocsvm.pkl", "rb"))
    th = json.load(open(p + "_threshold_ocsvm.json"))
    lists = [torch.load(p + f"_{k}_list_ocsvm.pt", map_location="cuda") for k in ("mean", "precision", "fea")]
    return dict(model=m, clr=clr, gmm=gmm, x0=th[0], lists=lists, path=p)


def ood_weight(ood, ood_tok, text):
    """eval_quant_o3_clinc.py 평가 루프와 같은 계산 (batch 1)."""
    x = ood_tok([text], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    mah = ood["model"].get_unsup_Mah_score_s(x, *ood["lists"])[:, 1:]
    s = ood["clr"].score_samples(mah)
    return max(0, q.obtain_weights(s, ood["gmm"], ood["x0"]))


@torch.no_grad()
def gen(model, tok, prompt, new, fixed):
    enc = tok([prompt], padding=True, return_tensors="pt", add_special_tokens=False)
    kw = dict(min_new_tokens=new) if fixed else {}
    out = model.generate(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(),
                         return_dict_in_generate=True, output_scores=False, max_new_tokens=new,
                         pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, **kw)
    return out.sequences.size(1) - enc["input_ids"].size(1)


def timed(fn, *a):
    sync(); t0 = time.perf_counter()
    r = fn(*a)
    sync()
    return time.perf_counter() - t0, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_gen", type=int, default=24)
    ap.add_argument("--n_natural", type=int, default=48)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    out = os.path.join(HERE, "results", "infer_o3_clinc.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    items = pick_items(a.n)
    texts = [str(x.get("instruction", "")) + (f"\n\n{x['input']}" if x.get("input") else "") for x in items]
    results = json.load(open(out)) if os.path.exists(out) else {}
    results["_meta"] = {"gpu": torch.cuda.get_device_name(0), "n_items": len(items), "reps": a.reps,
                        "natural_max_new_tokens": 128, "batch_size": 1}
    ood_tok = RobertaTokenizer.from_pretrained("roberta-large")
    for t in a.steps:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model, tok, lora = load_llm(t)
        r = {"t": t, "lora": lora, "llm_weights_MB": mb(torch.cuda.memory_allocated())}
        ood = load_ood(t)
        r["ood"] = ood["path"]
        r["with_ood_weights_MB"] = mb(torch.cuda.memory_allocated())
        r["ood_weights_MB"] = r["with_ood_weights_MB"] - r["llm_weights_MB"]
        prompts = [q.build_chat_prompt(tok, instruction=x, system=None) for x in texts]

        # OOD 점수 계산
        torch.cuda.reset_peak_memory_stats()
        for x in texts[:2]:
            ood_weight(ood, ood_tok, x)
        sec, ws = [], []
        for x in texts:
            s, w = timed(ood_weight, ood, ood_tok, x)
            sec.append(s); ws.append(float(w))
        r["ood_score_ms_per_sample"] = 1e3 * float(np.mean(sec))
        r["ood_score_ms_std"] = 1e3 * float(np.std(sec))
        r["ood_weight_mean"] = float(np.mean(ws))
        r["ood_peak_MB"] = mb(torch.cuda.max_memory_allocated())
        print(f"[o3 t{t}] OOD score {r['ood_score_ms_per_sample']:.1f} ms/sample, peak {r['ood_peak_MB']:.0f} MB", flush=True)

        # LLM 생성 (각 문항의 OOD weight 를 넣고, 원래 평가처럼)
        torch.cuda.reset_peak_memory_stats()
        for k in range(2):
            model.init_oodweight(ood_weight=[1, ws[k]]); gen(model, tok, prompts[k], 64, True)
        f_ms, p_ms = [], []
        for _ in range(a.reps):
            tf = tp = 0.0
            for k in range(a.n_gen):
                model.init_oodweight(ood_weight=[1, ws[k]])
                tf += timed(gen, model, tok, prompts[k], 64, True)[0]
                tp += timed(gen, model, tok, prompts[k], 1, False)[0]
            f_ms.append(1e3 * tf / a.n_gen); p_ms.append(1e3 * tp / a.n_gen)
        r["bs1"] = {"fixed64_ms_per_sample": f_ms, "prefill_ms_per_sample": p_ms,
                    "decode_ms_per_token": [(x - y) / 63 for x, y in zip(f_ms, p_ms)],
                    "fixed64_plus_ood_ms_per_sample": [x + r["ood_score_ms_per_sample"] for x in f_ms],
                    "peak_MB": mb(torch.cuda.max_memory_allocated())}
        print(f"[o3 t{t}] bs1 fixed64 {np.mean(f_ms):.1f} ms (+OOD {r['ood_score_ms_per_sample']:.1f}), "
              f"prefill {np.mean(p_ms):.1f}, decode {np.mean(r['bs1']['decode_ms_per_token']):.2f} ms/tok", flush=True)
        torch.cuda.reset_peak_memory_stats()
        tn, ntok = 0.0, []
        for k in range(a.n_natural):
            model.init_oodweight(ood_weight=[1, ws[k]])
            s, n = timed(gen, model, tok, prompts[k], 128, False)
            tn += s; ntok.append(n)
        r["natural"] = {"llm_ms_per_sample": 1e3 * tn / a.n_natural, "mean_new_tokens": float(np.mean(ntok)),
                        "total_ms_per_sample_incl_ood": 1e3 * tn / a.n_natural + r["ood_score_ms_per_sample"],
                        "peak_MB": mb(torch.cuda.max_memory_allocated())}
        print(f"[o3 t{t}] natural {r['natural']['llm_ms_per_sample']:.1f} ms/sample ({np.mean(ntok):.1f} tok) "
              f"+ OOD → {r['natural']['total_ms_per_sample_incl_ood']:.1f}", flush=True)
        results[f"o3_t{t}"] = r
        json.dump(results, open(out, "w"), indent=1)
        del model, ood
        torch.cuda.empty_cache()
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
