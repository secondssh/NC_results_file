#!/usr/bin/env python3
"""torch / transformers 버전이 생성 latency 에 주는 영향 실측: 같은 base Llama, 같은 prompt 토큰, 같은 생성 설정을
lunar (torch 2.5.1 / transformers 4.46.3) 와 o3 env (torch 2.1.1 / transformers 4.37.2) 에서 각각 잰다.
→ CLINC O3 LLM LoRA 의 학습 환경(torch / transformers 기록 없음, peft 0.18.1 만 확인) 때문에 생길 수 있는 오차의 크기.

두 env 모두 HF 기본 코드만 사용 (repo 코드 import 안 함): AutoModelForCausalLM bf16, greedy, use_cache.
attention: sdpa (RULE / base 와 같음) / eager (O3 modeling_llama_hacked_31 과 같은 방식) 둘 다.
입력: --make_prompts 로 lunar 에서 measure_infer_v2.py 와 같은 200 문항의 prompt 토큰을 json 으로 저장 → 두 env 가 같은 파일 사용.
측정: 64 토큰 고정 (min_new_tokens = max_new_tokens = 64), bs1 × 앞 24 문항 3 회 / bs16 × 200 문항 3 회, warm-up 2.
  요청마다 sync, TTFT = 첫 logits 시점 (LogitsProcessor), decode ms/token = (전체 - TTFT) / 63.
출력: latency/results/v2/version_{tag}.json
사용:
  lunar : python latency/version_check.py --make_prompts      (modify_code 에서, 1 회)
  각 env: CUDA_VISIBLE_DEVICES=0 <env python> latency/version_check.py --tag lunar|o3
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
BASE = os.path.join(MC, "train31Llama/trained_model/O3_LLAMA31_ClincQA")
PROMPTS = os.path.join(HERE, "results", "v2", "version_prompts_clinc.json")


def make_prompts():
    sys.path.insert(0, os.path.join(MC, "repro_check"))
    import random
    import router_token_eval as rte  # noqa: E402
    sys.path.insert(0, os.path.join(MC, "multi_attack"))
    from make_cross_data import src_paths  # noqa: E402
    from transformers import AutoTokenizer
    spec = rte.DS["clinc"]
    util = __import__(spec["util"], fromlist=["x"])
    cfg = rte.make_cfg(spec)
    tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")  # llama31model.py 와 같은 tokenizer
    src = src_paths("clinc", 8)
    rng = random.Random(1234)
    prompts = []
    for path, target in ((src["F"], "forget_edge"), (src["R"], "retained_edge")):
        recs, _ = rte.prompt_records(util, cfg, tok, path, target)
        prompts += [recs[i][0] for i in sorted(rng.sample(range(len(recs)), 100))]
    rng.shuffle(prompts)
    os.makedirs(os.path.dirname(PROMPTS), exist_ok=True)
    json.dump({"pad_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
               "prompts": prompts}, open(PROMPTS, "w"))
    print(f"[prompts] {len(prompts)} → {PROMPTS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_prompts", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--n_bs1", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results", "v2"))
    a = ap.parse_args()
    if a.make_prompts:
        return make_prompts()

    import transformers
    from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList

    class FirstToken(LogitsProcessor):
        def __init__(self):
            self.t = None

        def __call__(self, input_ids, scores):
            if self.t is None:
                torch.cuda.synchronize(); self.t = time.perf_counter()
            return scores

    d = json.load(open(PROMPTS))
    prompts, pad_id = d["prompts"], d["pad_id"]
    res = {"_meta": {"tag": a.tag, "torch": torch.__version__, "transformers": transformers.__version__,
                     "gpu": torch.cuda.get_device_name(0), "n_prompts": len(prompts), "reps": a.reps}}

    def one(model, P):
        S = max(map(len, P))
        ids = torch.tensor([[pad_id] * (S - len(p)) + p for p in P]).cuda()
        am = torch.tensor([[0] * (S - len(p)) + [1] * len(p) for p in P]).cuda()
        ft = FirstToken()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(input_ids=ids, attention_mask=am, max_new_tokens=64, min_new_tokens=64, do_sample=False,
                           num_beams=1, use_cache=True, pad_token_id=pad_id, logits_processor=LogitsProcessorList([ft]))
        torch.cuda.synchronize(); t1 = time.perf_counter()
        return 1e3 * (t1 - t0), 1e3 * (ft.t - t0), 1e3 * (t1 - ft.t) / 63

    for attn in ("sdpa", "eager"):
        model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16, attn_implementation=attn).cuda().eval()
        for bs, P in ((1, prompts[:a.n_bs1]), (16, prompts)):
            bl = [P[k:k + bs] for k in range(0, len(P), bs)]
            for b in bl[:2]:
                one(model, b)
            torch.cuda.reset_peak_memory_stats()
            recs = [one(model, b) for _ in range(a.reps) for b in bl]
            tot, ttft, dec = (np.array([r[i] for r in recs]) for i in range(3))
            key = f"{attn}_bs{bs}"
            res[key] = {"ms_per_sample": float(tot.sum() / a.reps / len(P)), "ms_per_request": float(tot.mean()),
                        "ms_per_request_std": float(tot.std()), "ttft_ms": float(ttft.mean()), "ttft_ms_std": float(ttft.std()),
                        "decode_ms_per_token": float(dec.mean()), "decode_std": float(dec.std()),
                        "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20}
            print(f"[{a.tag} {key}] {res[key]['ms_per_sample']:.1f} ms/sample, TTFT {res[key]['ttft_ms']:.2f}, "
                  f"decode {res[key]['decode_ms_per_token']:.2f} ms/tok, peak {res[key]['peak_allocated_MB']:.0f} MB", flush=True)
        del model; torch.cuda.empty_cache()
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"version_{a.tag}.json")
    json.dump(res, open(out, "w"), indent=1)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
