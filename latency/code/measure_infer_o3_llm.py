#!/usr/bin/env python3
"""O3 LLM 생성 latency + 메모리 (CLINC) — LLM LoRA 를 저장한 peft 버전 (0.18.1) 으로: lunar + PYTHONPATH peft0181.

모델 구성 / 생성은 quant_attack/eval_quant_o3_clinc.py (= eval_o3_clinc.py, quant_mode none) 와 같다:
  LlamaForCausalLM_ood (src/modeling_llama_hacked_31.py, eager attention) + PeftModel(hacked, 마지막 stage adapter, 'default')
  → 문항마다 model.init_oodweight([1, w]) → generate (bs1, greedy).
OOD weight w 는 measure_o3_ood.py (o3 env, detector 저장 환경) 가 같은 문항에 대해 계산·저장한 값을 그대로 쓴다
  → 실제 O3 와 같은 LoRA 세기로 생성. O3 1 문항 = OOD 점수 계산 (o3_ood_clinc_t{t}.json) + 이 LLM 생성.
입력: measure_infer_v2.py 와 같은 200 문항 / 같은 순서. prompt = 원래 평가처럼 build_chat_prompt(instruction).
설정 (warm-up 2 문항): fixed64 (min = max = 64) 앞 24 문항 3 회 / natural (O3 평가 설정 max_new_tokens = 128) 앞 48 문항 1 회.
측정: 요청마다 sync, TTFT = 첫 logits 시점 (LogitsProcessor), decode ms/token = (전체 - TTFT) / (step - 1).
출력: latency/results/v2/o3_llm_clinc_t{t}.json
사용 (ScienceQA_expriment 에서):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/hai_ssh/pylibs/peft0181 <lunar python> modify_code/latency/measure_infer_o3_llm.py --steps 0 8
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
REPO = os.path.dirname(MC)
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(1, HERE)
from measure_infer_o3 import CLINC_DOMAINS, load_llm, pick_items  # noqa: E402  (모델 구성 / 문항 선택은 기존 O3 측정과 같음)
import eval_quant_o3_clinc as q  # noqa: E402  (measure_infer_o3 가 path 를 잡음)
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402


class FirstToken(LogitsProcessor):
    def __init__(self):
        self.t = None

    def __call__(self, input_ids, scores):
        if self.t is None:
            torch.cuda.synchronize(); self.t = time.perf_counter()
        return scores


def stats(xs):
    a = np.asarray(xs, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "n": int(a.size)}


@torch.no_grad()
def one(model, tok, prompt, w, new, fixed):
    model.init_oodweight(ood_weight=[1, w])
    enc = tok([prompt], padding=True, return_tensors="pt", add_special_tokens=False)
    ids, am = enc["input_ids"].cuda(), enc["attention_mask"].cuda()
    ft = FirstToken()
    kw = dict(min_new_tokens=new) if fixed else {}
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = model.generate(input_ids=ids, attention_mask=am, return_dict_in_generate=True, output_scores=False,
                         max_new_tokens=new, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
                         logits_processor=LogitsProcessorList([ft]), **kw)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    steps = out.sequences.size(1) - ids.size(1)
    return {"ms": 1e3 * (t1 - t0), "ttft_ms": 1e3 * (ft.t - t0), "decode_ms_per_token": 1e3 * (t1 - ft.t) / max(steps - 1, 1),
            "steps": steps, "prompt_tokens": int(ids.size(1))}


def run(model, tok, prompts, ws, new, fixed, reps):
    for k in range(2):
        one(model, tok, prompts[k], ws[k], new, fixed)
    torch.cuda.reset_peak_memory_stats()
    recs = [one(model, tok, p, w, new, fixed) for _ in range(reps) for p, w in zip(prompts, ws)]
    return {"ms_per_sample": stats([r["ms"] for r in recs]), "ttft_ms": stats([r["ttft_ms"] for r in recs]),
            "decode_ms_per_token": stats([r["decode_ms_per_token"] for r in recs]),
            "mean_new_tokens": float(np.mean([r["steps"] for r in recs])),
            "mean_prompt_tokens": float(np.mean([r["prompt_tokens"] for r in recs])),
            "n_samples": len(prompts), "reps": reps, "warmup": 2, "bs": 1,
            "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 8])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_fixed", type=int, default=24)
    ap.add_argument("--n_natural", type=int, default=48)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results", "v2"))
    a = ap.parse_args()
    import peft
    import transformers
    items = pick_items(a.n)
    texts = [str(x.get("instruction", "")) + (f"\n\n{x['input']}" if x.get("input") else "") for x in items]
    for t in a.steps:
        ood = json.load(open(os.path.join(a.out_dir, f"o3_ood_clinc_t{t}.json")))
        ws = ood["ood_weights"]
        assert len(ws) == len(texts)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        model, tok, lora = load_llm(t)
        prompts = [q.build_chat_prompt(tok, instruction=x, system=None) for x in texts]
        res = {"t": t, "lora": lora, "env": {"torch": torch.__version__, "transformers": transformers.__version__,
                                             "peft": peft.__version__, "peft_path": peft.__file__},
               "gpu": torch.cuda.get_device_name(0), "llm_weights_MB": torch.cuda.memory_allocated() / 2**20,
               "attn": "eager (src/modeling_llama_hacked_31.py LlamaAttention)", "ood_weights_from": f"o3_ood_clinc_t{t}.json"}
        res["fixed64_bs1"] = run(model, tok, prompts[:a.n_fixed], ws[:a.n_fixed], 64, True, 3)
        res["natural_bs1"] = run(model, tok, prompts[:a.n_natural], ws[:a.n_natural], 128, False, 1)
        for k in ("fixed64_bs1", "natural_bs1"):
            r = res[k]
            r["ms_per_sample_incl_ood"] = r["ms_per_sample"]["mean"] + ood["total_ms"]["mean"]
            r["ttft_incl_ood_ms"] = r["ttft_ms"]["mean"] + ood["total_ms"]["mean"]
            print(f"[o3 llm t{t}] {k}: {r['ms_per_sample']['mean']:.1f} ms/sample (+OOD → {r['ms_per_sample_incl_ood']:.1f}), "
                  f"TTFT {r['ttft_ms']['mean']:.2f} (+OOD → {r['ttft_incl_ood_ms']:.2f}), decode {r['decode_ms_per_token']['mean']:.2f} ms/tok, "
                  f"{r['mean_new_tokens']:.1f} tok, peak {r['peak_allocated_MB']:.0f} MB", flush=True)
        out = os.path.join(a.out_dir, f"o3_llm_clinc_t{t}.json")
        json.dump(res, open(out, "w"), indent=1)
        del model; torch.cuda.empty_cache()
    print("[done]")


if __name__ == "__main__":
    main()
