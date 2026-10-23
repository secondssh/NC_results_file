#!/usr/bin/env python3
"""expert 수 E 에 따른 RULE 비용 (요청 80 건 이상). 연산량은 가중치 값과 무관 (top-1 이어도 모든 expert 의 LoRA 를 계산 후 섞음,
값에 따라 갈리는 분기 없음) → 학습 안 한 expert 로 E 를 늘려 측정. 출력 길이가 가중치에 좌우되므로 생성은 64 tok 고정만.

E = 2, 10, 21 : 실제 TOFU ckpt (t0, t8, t19)
E = 40 … 320 : TOFU t19 (실제 21 expert) + 랜덤 expert (expand_lora_moe_experts 와 같은 N(0, 0.01) 초기화, gate 행도 같은 방식)
  1) microbench : layer16 MoE 하나 (microbench_moe.py 와 같은 연산), decode [B,1] / prefill [B,50], B = 1 / 8 (TOFU eval batch),
                  호출당 µs (warm-up 50, 500 회) + 호출 1 회 peak 추가 메모리 (BAx [B,S,E,4096] 가 E 에 비례)
  2) e2e        : 실제 TOFU base 모델 layer16 down_proj 자리에 E-expert MoE, measure_infer_v2 와 같은 200 문항 / 같은 경로,
                  64 tok 고정 bs1 (앞 24 문항 × 3) + bs8 (200 문항 × 3), TTFT / decode / ms/sample / peak
  3) train step : MoE 학습 1 step (train_moe.py joint 와 같은 loss: 새 expert MSE + router CE, AdamW, 이전 expert grad 0),
                  batch 64 (원래 MOE_TRAIN_BS), 랜덤 activation, warm-up 5 후 50 step 평균, peak 메모리
  4) params     : E × (147,456 + 14,337)
출력: latency/results/v2/timing/scale_experts.json
사용 (modify_code 에서): CUDA_VISIBLE_DEVICES=0 python latency/scale_experts.py
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, MC)
from microbench_moe import build, router, expert, bench  # noqa: E402
import measure_infer_v2 as mi  # noqa: E402  (router_token_eval import → modify_code 로 chdir)
from src_lunar.estimated_net_moe_utils import LUNAR_LoRA_MoE_net  # noqa: E402

REAL = {2: 0, 10: 8, 21: 19}  # E → TOFU step
OUT = os.path.join(HERE, "results", "v2", "timing", "scale_experts.json")


def ckpt(t):
    return f"outputs/tofu_ckpt_repro_lunar/seed_0/t{t}_AG{t + 1:02d}/lora_moe.pt"


def make_moe(E, seed=0):
    if E in REAL:
        return build(ckpt(REAL[E]))
    src = build(ckpt(19))
    g = torch.Generator(device="cuda").manual_seed(seed)
    m = LUNAR_LoRA_MoE_net(input_dim=14336, output_dim=4096, n_experts=E, top_k=1, gate_hidden=0,
                           lora_rank=8, lora_alpha=8, lora_dropout=0.0, pretrained_weight=src.linear.weight.data).cuda().to(torch.bfloat16)
    with torch.no_grad():
        n = src.n_experts
        m.lora_A.normal_(0, 0.01, generator=g); m.lora_B.normal_(0, 0.01, generator=g)
        m.gate.weight.normal_(0, 0.01, generator=g); m.gate.bias.zero_()
        m.lora_A[:n].copy_(src.lora_A); m.lora_B[:n].copy_(src.lora_B)
        m.gate.weight[:n].copy_(src.gate.weight); m.gate.bias[:n].copy_(src.gate.bias)
    assert m.scaling == src.scaling, (m.scaling, src.scaling)
    del src; torch.cuda.empty_cache()
    return m.eval()


@torch.no_grad()
def peak_extra(fn):
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn(); torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2**20


def micro(m):
    out = {}
    for phase, S in (("decode", 1), ("prefill", 50)):
        for B in (1, 8):
            x = torch.randn(B, S, 14336, device="cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                probs, base = router(m, x), m.linear(x)
            out[f"{phase}_bs{B}"] = {"base_linear": bench(lambda: m.linear(x)), "router": bench(lambda: router(m, x)),
                                     "expert": bench(lambda: expert(m, x, probs, base)), "moe_full": bench(lambda: m(x)),
                                     "moe_extra_mem_MB": peak_extra(lambda: m(x)), "linear_extra_mem_MB": peak_extra(lambda: m.linear(x))}
            r = out[f"{phase}_bs{B}"]; r["added_vs_base"] = r["moe_full"] - r["base_linear"]
    return out


def train_step(m, steps=50, warm=5, bs=64):
    m.train()
    E = m.n_experts
    cur = E - 1
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=0.01)
    x = torch.randn(bs, 14336, device="cuda", dtype=torch.bfloat16)
    y = torch.randn(bs, 4096, device="cuda", dtype=torch.bfloat16)
    lab = torch.where(torch.rand(bs, device="cuda") < 0.5, torch.full((bs,), cur, device="cuda"), torch.zeros(bs, dtype=torch.long, device="cuda"))
    keep = torch.zeros(E, dtype=torch.bool, device="cuda"); keep[0] = True; keep[cur] = True

    def step():
        opt.zero_grad(set_to_none=True)
        y_mix, logits, BAx, base = m(x, return_router_logits=True, return_all_experts=True)
        fm = lab == cur
        mse = F.mse_loss(base[fm] + m.scaling * BAx[fm, cur], y[fm])
        ce = F.cross_entropy(logits.float().masked_fill(~keep, -1e9), lab)
        (mse + 0.5 * ce).backward()
        with torch.no_grad():  # train_moe.py: retain + 이전 expert / gate 행 grad 0
            m.lora_A.grad[:cur].zero_(); m.lora_B.grad[:cur].zero_()
            m.gate.weight.grad[1:cur].zero_(); m.gate.bias.grad[1:cur].zero_()
        opt.step()

    for _ in range(warm):
        step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    r = {"ms_per_step": 1e3 * (time.perf_counter() - t0) / steps, "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
         "batch": bs, "steps": steps}
    m.eval()
    del opt; torch.cuda.empty_cache()
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, nargs="+", default=[2, 10, 21, 40, 80, 160, 320])
    ap.add_argument("--e2e_E", type=int, nargs="+", default=[21, 80, 160, 320])
    ap.add_argument("--n_bs1", type=int, default=24)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res["_meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "real_E": REAL,
                    "synthetic": "TOFU t19 21 experts + random N(0,0.01) experts", "per_expert_params": 147456 + 14337}

    def save():
        json.dump(res, open(OUT, "w"), indent=1)

    # 1) microbench + 3) train step + 4) params (모듈만)
    for E in a.E:
        m = make_moe(E)
        r = {"E": E, "real": E in REAL, "params": E * (147456 + 14337),
             "module_weights_MB": sum(p.numel() * p.element_size() for n, p in m.named_parameters() if n != "linear.weight") / 2**20}
        r["micro"] = micro(m)
        r["train_step"] = train_step(m)
        res[f"E{E}"] = {**res.get(f"E{E}", {}), **r}; save()
        mb = r["micro"]
        print(f"[E={E}] params {r['params'] / 1e6:.2f}M | decode bs1 +{mb['decode_bs1']['added_vs_base']:.0f}us bs8 +{mb['decode_bs8']['added_vs_base']:.0f}us | "
              f"prefill bs1 +{mb['prefill_bs1']['added_vs_base']:.0f}us bs8 +{mb['prefill_bs8']['added_vs_base']:.0f}us "
              f"(extra mem {mb['prefill_bs8']['moe_extra_mem_MB']:.0f}MB) | train step {r['train_step']['ms_per_step']:.1f}ms "
              f"peak {r['train_step']['peak_allocated_MB']:.0f}MB", flush=True)
        del m; gc.collect(); torch.cuda.empty_cache()

    # 2) e2e: 실제 base 모델 + E-expert MoE (measure_infer_v2 와 같은 문항 / 경로)
    spec = mi.rte.DS["tofu"]
    util = __import__(spec["util"], fromlist=["x"])
    cfg = mi.rte.make_cfg(spec)
    model = mi.load_model(cfg.model_family, cfg.model_path, torch.device("cuda"))
    model.model.eval()
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    src = mi.src_paths("tofu", 19)
    rng = random.Random(1234)
    prompts = []
    for path, target in ((src["F"], "forget_edge"), (src["R"], "retained_edge")):
        recs, _ = mi.rte.prompt_records(util, cfg, tok, path, target)
        prompts += [recs[i][0] for i in sorted(rng.sample(range(len(recs)), 100))]
    rng.shuffle(prompts)
    layer = model.model_block_modules[16].mlp
    orig = layer.down_proj
    for E in ["base"] + a.e2e_E:
        if E == "base":
            layer.down_proj = orig
        else:
            layer.down_proj = make_moe(E)
        gc.collect(); torch.cuda.empty_cache()
        r = {"weights_MB": torch.cuda.memory_allocated() / 2**20}
        r["fixed64_bs1"] = mi.run(model, prompts[:a.n_bs1], 1, pad_id, 64, True, 3)
        r["fixed64_bs8"] = mi.run(model, prompts, 8, pad_id, 64, True, 3)
        for k in ("fixed64_bs1", "fixed64_bs8"):
            r[k].pop("per_request", None)
        key = "e2e_base" if E == "base" else f"E{E}"
        res[key] = {**res.get(key, {}), "e2e": r} if E != "base" else {"e2e": r}; save()
        print(f"[e2e {E}] bs1 {r['fixed64_bs1']['ms_per_sample']['mean']:.1f} ms, TTFT {r['fixed64_bs1']['ttft_ms']['mean']:.2f}, "
              f"decode {r['fixed64_bs1']['decode_ms_per_token']['mean']:.2f} | bs8 {r['fixed64_bs8']['ms_per_sample']['mean']:.1f} ms, "
              f"TTFT {r['fixed64_bs8']['ttft_ms']['mean']:.2f}, peak {r['fixed64_bs8']['mem']['peak_allocated_MB']:.0f} MB", flush=True)
    layer.down_proj = orig
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
