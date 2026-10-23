#!/usr/bin/env python3
"""추론 latency + GPU 메모리: base(MoE 없음) vs RULE(시점별). README "### latency 실험" 1).

입력: 데이터셋마다 고정 200 문항 (마지막 시점 SD 에서 forget 100 + RD 에서 retain 100, seed 1234)
      → base 와 모든 시점 RULE 에 같은 prompt (원래 eval 과 같은 dataloader → chat template prompt 토큰)
생성: 원래 eval 과 같은 경로 (left pad, model.model.generate, greedy, use_cache)
  fixed   : min_new_tokens = max_new_tokens = 64  → 모델 간 비교 (RULE 거절문 / label 길이 차이 제거)
  prefill : max_new_tokens = 1                    → decode ms/token = (fixed - prefill) / 63
  natural : 원래 설정 (max_new_tokens = cfg 512)  → 실사용 기준, eval batch 만
batch: 원래 eval batch (CLINC 16 / TOFU 8) 200 문항 전부 + bs=1 앞 24 문항.  warmup 2 batch, reps 회 반복.
메모리: 모델(+MoE) 올린 뒤 memory_allocated (가중치), 설정마다 reset 후 max_memory_allocated / max_memory_reserved.
분해 (RULE, 별도 pass, fixed eval batch 1 회): layer16 LUNAR_LoRA_MoE_net.forward 를 같은 계산으로 감싸 CUDA event 로
  base linear (원래 down_proj) / router (gate + softmax + argmax + topk mask) / expert (LoRA einsum 3개 + 섞기) 시간 합.
  base 모델은 layer16 down_proj(nn.Linear) 시간만 같은 방식으로.

출력: latency/results/infer_{ds}.json
사용 (modify_code 에서): CUDA_VISIBLE_DEVICES=0 python latency/measure_infer.py --ds clinc --steps 0 4 8
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
import types

import numpy as np
import torch
import torch.nn.functional as F

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


def sync():
    torch.cuda.synchronize()


def batches(prompts, bs, pad_id):
    for k in range(0, len(prompts), bs):
        P = prompts[k:k + bs]
        S = max(map(len, P))
        ids = torch.tensor([[pad_id] * (S - len(p)) + p for p in P]).cuda()
        am = torch.tensor([[0] * (S - len(p)) + [1] * len(p) for p in P]).cuda()
        yield ids, am


@torch.no_grad()
def gen(model, ids, am, pad_id, new, fixed):
    kw = dict(min_new_tokens=new) if fixed else {}
    return model.model.generate(input_ids=ids, attention_mask=am, max_new_tokens=new, do_sample=False, num_beams=1,
                                num_return_sequences=1, use_cache=True, pad_token_id=pad_id,
                                output_scores=False, return_dict_in_generate=True, **kw)


def run(model, prompts, bs, pad_id, new, fixed, reps, warmup=2):
    """reps 회 전체 시간 (초) + 생성 토큰 수 (eos 전까지, 샘플 평균)."""
    bl = list(batches(prompts, bs, pad_id))
    for ids, am in bl[:warmup]:
        gen(model, ids, am, pad_id, new, fixed)
    sync()
    times, ntok = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        n = []
        for ids, am in bl:
            out = gen(model, ids, am, pad_id, new, fixed)
            seq = out.sequences[:, ids.size(1):]
            n += [int((r != pad_id).sum()) for r in seq] if not fixed else [new] * seq.size(0)
        sync()
        times.append(time.perf_counter() - t0)
        ntok = n
    return {"sec": times, "n_batches": len(bl), "n_samples": len(prompts), "mean_new_tokens": float(np.mean(ntok))}


class Timer:
    """forward 안에서 CUDA event 쌍을 쌓아 두고 끝난 뒤 합산 (sync 없음)."""

    def __init__(self):
        self.ev = {}

    def mark(self, key):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.ev.setdefault(key, []).append(e)

    def totals(self, pairs):
        sync()
        out = {}
        for name, (a, b) in pairs.items():
            s = self.ev.get(a, []); e = self.ev.get(b, [])
            out[name] = float(sum(x.elapsed_time(y) for x, y in zip(s, e)))
        out["n_calls"] = len(self.ev.get("start", []))
        return out


def timed_moe_forward(timer):
    """estimated_net_moe_utils.LUNAR_LoRA_MoE_net.forward 와 같은 계산 + 구간 event."""
    def forward(self, x, attention_mask=None, return_gate=False, return_router_logits=False,
                return_all_experts=False, forced_expert_idx=None):
        timer.mark("start")
        x = self._ensure_dtype(x)
        base = self.linear(x)
        timer.mark("linear_end")
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        top1 = probs.argmax(dim=-1)  # noqa: F841 (원래 코드와 같은 계산)
        if self.top_k < self.n_experts:
            _, topi = torch.topk(probs, k=self.top_k, dim=-1)
            mask = torch.zeros_like(probs)
            mask.scatter_(-1, topi, 1.0)
            probs = probs * mask
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
        timer.mark("router_end")
        x_d = self.lora_dropout(x)
        Ax = torch.einsum("...h,erh->...er", x_d, self.lora_A)
        BAx = torch.einsum("...er,eor->...eo", Ax, self.lora_B)
        update = torch.einsum("...e,...eo->...o", probs, BAx)
        out = base + self.scaling * update
        timer.mark("expert_end")
        return out
    return forward


def timed_linear_forward(timer, orig):
    def forward(x):
        timer.mark("start")
        y = orig(x)
        timer.mark("linear_end")
        return y
    return forward


def mem_snapshot():
    return {"allocated_MB": torch.cuda.memory_allocated() / 2**20,
            "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20}


def measure(model, tag, prompts, eval_bs, pad_id, cfg, reps, moe, n_bs1):
    res = {"tag": tag}
    gc.collect(); torch.cuda.empty_cache()
    res["weights_MB"] = torch.cuda.memory_allocated() / 2**20
    for bs, P in ((eval_bs, prompts), (1, prompts[:n_bs1])):
        torch.cuda.reset_peak_memory_stats()
        f = run(model, P, bs, pad_id, 64, True, reps)
        p = run(model, P, bs, pad_id, 1, False, reps)
        ms_f = [1e3 * s / f["n_batches"] for s in f["sec"]]
        ms_p = [1e3 * s / p["n_batches"] for s in p["sec"]]
        res[f"bs{bs}"] = {
            "fixed64_ms_per_batch": ms_f, "prefill_ms_per_batch": ms_p,
            "decode_ms_per_token": [(a - b) / 63 for a, b in zip(ms_f, ms_p)],
            "fixed64_ms_per_sample": [1e3 * s / f["n_samples"] for s in f["sec"]],
            "n_batches": f["n_batches"], "n_samples": f["n_samples"], "mem": mem_snapshot()}
        print(f"[{tag}] bs{bs}: fixed64 {np.mean(ms_f):.1f} ms/batch, prefill {np.mean(ms_p):.1f}, "
              f"decode {np.mean(res[f'bs{bs}']['decode_ms_per_token']):.2f} ms/tok, peak {res[f'bs{bs}']['mem']['peak_allocated_MB']:.0f} MB", flush=True)
    torch.cuda.reset_peak_memory_stats()
    n = run(model, prompts, eval_bs, pad_id, int(cfg.eval_generation_max_new_tokens), False, max(1, reps - 1))
    res["natural"] = {"ms_per_sample": [1e3 * s / n["n_samples"] for s in n["sec"]], "mean_new_tokens": n["mean_new_tokens"],
                      "bs": eval_bs, "mem": mem_snapshot()}
    print(f"[{tag}] natural bs{eval_bs}: {np.mean(res['natural']['ms_per_sample']):.1f} ms/sample, "
          f"{n['mean_new_tokens']:.1f} tok, peak {res['natural']['mem']['peak_allocated_MB']:.0f} MB", flush=True)

    # 분해: 같은 fixed64 eval batch 1 회, 구간 event
    timer = Timer()
    if moe is not None:
        orig = moe.forward
        moe.forward = types.MethodType(timed_moe_forward(timer), moe)
    else:
        lin = model.model_block_modules[rte.LAYER].mlp.down_proj
        orig = lin.forward
        lin.forward = timed_linear_forward(timer, orig)
    t0 = time.perf_counter()
    for ids, am in batches(prompts, eval_bs, pad_id):
        gen(model, ids, am, pad_id, 64, True)
    sync()
    wall = 1e3 * (time.perf_counter() - t0)
    pairs = {"linear_ms": ("start", "linear_end")}
    if moe is not None:
        pairs.update({"router_ms": ("linear_end", "router_end"), "expert_ms": ("router_end", "expert_end"),
                      "layer16_total_ms": ("start", "expert_end")})
    else:
        pairs["layer16_total_ms"] = ("start", "linear_end")
    br = timer.totals(pairs)
    br["pass_wall_ms"] = wall
    res["breakdown_fixed64_evalbs"] = br
    if moe is not None:
        moe.forward = orig
    else:
        del lin.forward
    print(f"[{tag}] breakdown: " + ", ".join(f"{k} {v:.1f}" for k, v in br.items()), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["clinc", "tofu"])
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100, help="forget / retain 각각 문항 수")
    ap.add_argument("--n_bs1", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no_base", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "results", f"infer_{a.ds}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    spec = rte.DS[a.ds]
    util = __import__(spec["util"], fromlist=["x"])
    cfg = rte.make_cfg(spec)
    eval_bs = int(cfg.eval_batch_size)
    model = load_model(cfg.model_family, cfg.model_path, torch.device("cuda"))
    model.model.eval()
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    src = src_paths(a.ds, LAST[a.ds])
    rng = random.Random(1234)
    prompts = []
    for path, target in ((src["F"], "forget_edge"), (src["R"], "retained_edge")):
        recs, _ = rte.prompt_records(util, cfg, tok, path, target)
        prompts += [recs[i][0] for i in sorted(rng.sample(range(len(recs)), a.n))]
    rng.shuffle(prompts)  # forget / retain 섞인 batch (원래 eval 은 split 별이지만 latency 는 내용 무관)

    meta = {"ds": a.ds, "eval_bs": eval_bs, "n_prompts": len(prompts), "mean_prompt_tokens": float(np.mean([len(p) for p in prompts])),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "reps": a.reps,
            "max_new_tokens_natural": int(cfg.eval_generation_max_new_tokens)}
    results = json.load(open(out)) if os.path.exists(out) else {}
    results["_meta"] = meta
    if not a.no_base:
        results["base"] = measure(model, f"{a.ds} base", prompts, eval_bs, pad_id, cfg, a.reps, None, a.n_bs1)
        json.dump(results, open(out, "w"), indent=1)
    for t in a.steps:
        with contextlib.redirect_stdout(io.StringIO()):
            load_lora_moe_only(model=model, load_path=ckpt(a.ds, t), layer_idxs=[rte.LAYER], device="cuda", dtype=torch.bfloat16)
        moe = model.model_block_modules[rte.LAYER].mlp.down_proj
        assert int(moe.n_experts) == t + 2
        r = measure(model, f"{a.ds} RULE t{t}", prompts, eval_bs, pad_id, cfg, a.reps, moe, a.n_bs1)
        r["n_experts"] = t + 2
        results[f"rule_t{t}"] = r
        json.dump(results, open(out, "w"), indent=1)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
