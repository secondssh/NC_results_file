#!/usr/bin/env python3
"""추론 latency v2: base / RULE (시점별) / GD. measure_infer.py 와 같은 모델·입력·생성 경로, 측정 방식만 보강.

원래 평가와 같게 유지: lunar env, bf16, HF AutoModelForCausalLM (기본 SDPA), chat template prompt (원래 dataloader),
  greedy, use_cache, left pad, eval batch (CLINC 16 / TOFU 8), max_new_tokens = cfg 512.
입력: 데이터셋마다 같은 200 문항 (마지막 시점 SD forget 100 + RD retain 100, seed 1234, measure_infer.py 와 동일).
설정 (warm-up 2 batch 후):
  natural (주 결과, 원래 생성 설정) : eval batch × 200 문항 2 회, bs1 × 앞 24 문항 1 회
  fixed64 (보조, 연산 비교용)      : min_new_tokens = max_new_tokens = 64, eval batch × 200 문항 3 회, bs1 × 앞 24 문항 3 회
측정: 요청(batch)마다 sync → generate → sync (perf_counter).
  TTFT = 같은 생성 안에서 첫 logits 가 나온 시점 (값을 바꾸지 않는 LogitsProcessor, 첫 호출에만 sync)
  decode ms/token = (전체 - TTFT) / (생성 step 수 - 1),  step 수 = batch 안 가장 긴 생성 길이
  메모리 = 설정마다 reset 후 max_memory_allocated / max_memory_reserved (+ monitor.py 로 nvidia-smi / host RAM)
출력: latency/results/v2/infer_{ds}.json
사용 (modify_code 에서):
  CUDA_VISIBLE_DEVICES=0 python latency/measure_infer_v2.py --ds clinc --steps 0 4 8 --gd_path <GD ckpt dir>
"""
import argparse
import contextlib
import gc
import io
import json
import os
import random
import sys
import time

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MC, "repro_check"))
import router_token_eval as rte  # noqa: E402  (import 시 modify_code 로 chdir)

sys.path.insert(0, os.path.join(MC, "multi_attack"))
from make_cross_data import src_paths  # noqa: E402
from router_cross import ckpt  # noqa: E402

from src_lunar.model_utils.model_loader import load_model  # noqa: E402
from src_lunar.estimated_net_moe_utils import load_lora_moe_only  # noqa: E402

LAST = {"clinc": 8, "tofu": 19}


class FirstToken(LogitsProcessor):
    """첫 호출(= prefill forward 끝, 첫 토큰 logits 준비) 시각만 기록. scores 는 그대로 반환."""

    def __init__(self):
        self.t = None

    def __call__(self, input_ids, scores):
        if self.t is None:
            torch.cuda.synchronize()
            self.t = time.perf_counter()
        return scores


def batches(prompts, bs, pad_id):
    for k in range(0, len(prompts), bs):
        P = prompts[k:k + bs]
        S = max(map(len, P))
        ids = torch.tensor([[pad_id] * (S - len(p)) + p for p in P]).cuda()
        am = torch.tensor([[0] * (S - len(p)) + [1] * len(p) for p in P]).cuda()
        yield ids, am


@torch.no_grad()
def one(model, ids, am, pad_id, new, fixed):
    ft = FirstToken()
    kw = dict(min_new_tokens=new) if fixed else {}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.model.generate(input_ids=ids, attention_mask=am, max_new_tokens=new, do_sample=False, num_beams=1,
                               num_return_sequences=1, use_cache=True, pad_token_id=pad_id, output_scores=False,
                               return_dict_in_generate=True, logits_processor=LogitsProcessorList([ft]), **kw)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    steps = out.sequences.size(1) - ids.size(1)
    new_tok = [int((r != pad_id).sum()) for r in out.sequences[:, ids.size(1):]]
    return {"ms": 1e3 * (t1 - t0), "ttft_ms": 1e3 * (ft.t - t0),
            "decode_ms_per_token": 1e3 * (t1 - ft.t) / max(steps - 1, 1), "steps": steps,
            "bs": int(ids.size(0)), "mean_new_tokens": float(np.mean(new_tok))}


def stats(xs):
    a = np.asarray(xs, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "n": int(a.size)}


def run(model, prompts, bs, pad_id, new, fixed, reps, warmup=2):
    bl = list(batches(prompts, bs, pad_id))
    for ids, am in bl[:warmup]:
        one(model, ids, am, pad_id, new, fixed)
    torch.cuda.reset_peak_memory_stats()
    recs = []
    for r in range(reps):
        for ids, am in bl:
            x = one(model, ids, am, pad_id, new, fixed); x["rep"] = r
            recs.append(x)
    n_samples = len(prompts)
    tot = [sum(x["ms"] for x in recs if x["rep"] == r) for r in range(reps)]
    return {"per_request": recs,
            "ms_per_request": stats([x["ms"] for x in recs]),
            "ms_per_sample": stats([t / n_samples for t in tot]),
            "ttft_ms": stats([x["ttft_ms"] for x in recs]),
            "decode_ms_per_token": stats([x["decode_ms_per_token"] for x in recs]),
            "mean_new_tokens": float(np.mean([x["mean_new_tokens"] for x in recs])),
            "mean_steps": float(np.mean([x["steps"] for x in recs])),
            "bs": bs, "n_samples": n_samples, "reps": reps, "warmup_batches": warmup,
            "mem": {"peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20}}


def measure(model, tag, prompts, eval_bs, pad_id, max_new, n_bs1):
    gc.collect(); torch.cuda.empty_cache()
    res = {"tag": tag, "weights_MB": torch.cuda.memory_allocated() / 2**20}
    plan = [("natural_evalbs", eval_bs, prompts, max_new, False, 2), ("natural_bs1", 1, prompts[:n_bs1], max_new, False, 1),
            ("fixed64_evalbs", eval_bs, prompts, 64, True, 3), ("fixed64_bs1", 1, prompts[:n_bs1], 64, True, 3)]
    for name, bs, P, new, fixed, reps in plan:
        r = run(model, P, bs, pad_id, new, fixed, reps)
        res[name] = r
        print(f"[{tag}] {name}: {r['ms_per_sample']['mean']:.1f} ms/sample, TTFT {r['ttft_ms']['mean']:.2f}±{r['ttft_ms']['std']:.2f} ms, "
              f"decode {r['decode_ms_per_token']['mean']:.2f} ms/tok, {r['mean_new_tokens']:.1f} tok, "
              f"peak {r['mem']['peak_allocated_MB']:.0f} MB", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["clinc", "tofu"])
    ap.add_argument("--steps", type=int, nargs="*", default=[])
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--gd_path", default=None, help="GD(ga_gd) full-FT ckpt 폴더 (구조 = base, 가중치만 다름)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_bs1", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "results", "v2", f"infer_{a.ds}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    spec = rte.DS[a.ds]
    util = __import__(spec["util"], fromlist=["x"])
    cfg = rte.make_cfg(spec)
    eval_bs, max_new = int(cfg.eval_batch_size), int(cfg.eval_generation_max_new_tokens)
    results = json.load(open(out)) if os.path.exists(out) else {}

    def load(path):
        m = load_model(cfg.model_family, path, torch.device("cuda"))
        m.model.eval()
        return m

    model = load(cfg.model_path)
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    src = src_paths(a.ds, LAST[a.ds])
    rng = random.Random(1234)
    prompts = []
    for path, target in ((src["F"], "forget_edge"), (src["R"], "retained_edge")):
        recs, _ = rte.prompt_records(util, cfg, tok, path, target)
        prompts += [recs[i][0] for i in sorted(rng.sample(range(len(recs)), a.n))]
    rng.shuffle(prompts)
    results["_meta"] = {"ds": a.ds, "eval_bs": eval_bs, "max_new_tokens_natural": max_new, "n_prompts": len(prompts),
                        "mean_prompt_tokens": float(np.mean([len(p) for p in prompts])),
                        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                        "attn_implementation": getattr(model.model.config, "_attn_implementation", None),
                        "dtype": str(next(model.model.parameters()).dtype),
                        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32}

    def save():
        json.dump(results, open(out, "w"), indent=1)

    if not a.no_base:
        results["base"] = measure(model, f"{a.ds} base", prompts, eval_bs, pad_id, max_new, a.n_bs1); save()
    for t in a.steps:
        with contextlib.redirect_stdout(io.StringIO()):
            load_lora_moe_only(model=model, load_path=ckpt(a.ds, t), layer_idxs=[rte.LAYER], device="cuda", dtype=torch.bfloat16)
        assert int(model.model_block_modules[rte.LAYER].mlp.down_proj.n_experts) == t + 2
        r = measure(model, f"{a.ds} RULE t{t}", prompts, eval_bs, pad_id, max_new, a.n_bs1)
        r["n_experts"] = t + 2
        results[f"rule_t{t}"] = r; save()
    if a.gd_path:
        del model; gc.collect(); torch.cuda.empty_cache()
        gd = load(a.gd_path)
        r = measure(gd, f"{a.ds} GD", prompts, eval_bs, pad_id, max_new, a.n_bs1)
        r["model_path"] = a.gd_path
        results["gd"] = r; save()
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
