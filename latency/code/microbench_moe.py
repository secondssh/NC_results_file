#!/usr/bin/env python3
"""Router / Expert 계산 microbenchmark: layer16 MoE 하나만 떼서 같은 입력 크기로 반복 측정. README "### latency 실험" 4).

모듈: 실제 RULE ckpt 로 LUNAR_LoRA_MoE_net 생성 (load_lora_moe_only 와 같은 방식, bf16).
입력: x ~ N(0,1) bf16, shape [B, S, 14336] (계산량은 값과 무관, RULE 은 top-1 이어도 모든 expert 의 LoRA 를 계산)
  decode  : S=1,  B = 1 / eval batch (CLINC 16, TOFU 8)
  prefill : S = 평균 prompt 길이 (CLINC 64, TOFU 50), B = 1 / eval batch
측정 (각각 warm-up 50 회 후 500 회, CUDA event, 호출당 µs):
  base_linear : 원래 down_proj (nn.Linear 14336→4096, 이것은 base 모델에도 있음)
  router      : gate Linear(14336→E) + softmax + argmax + topk mask + normalize   (forward 와 같은 연산)
  expert      : dropout + einsum 3 개 (A·x, B·Ax, probs 로 섞기) + base + scaling·update
  moe_full    : LUNAR_LoRA_MoE_net.forward 전체
  추가 비용 = moe_full - base_linear (= base 모델 대비 layer16 에서 늘어나는 시간, 32 layer 중 1 개)
출력: latency/results/v2/microbench_moe.json
사용 (modify_code 에서): CUDA_VISIBLE_DEVICES=0 python latency/microbench_moe.py
"""
import json
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
sys.path.insert(0, MC)
os.chdir(MC)
from src_lunar.estimated_net_moe_utils import LUNAR_LoRA_MoE_net  # noqa: E402

CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]
SETS = {"clinc": dict(steps=[0, 4, 8], bs=16, S=64), "tofu": dict(steps=[0, 10, 19], bs=8, S=50)}
WARM, ITERS = 50, 500


def ckpt(ds, t):
    if ds == "clinc":
        return f"outputs/repro_clinc_lunar_bs64/seed_0/t{t}_{CLINC_DOMAINS[t]}/lora_moe.pt"
    return f"outputs/tofu_ckpt_repro_lunar/seed_0/t{t}_AG{t + 1:02d}/lora_moe.pt"


def build(path):
    obj = torch.load(path, map_location="cpu", weights_only=False)[16]
    dev, dt = "cuda", torch.bfloat16
    m = LUNAR_LoRA_MoE_net(input_dim=int(obj["input_dim"]), output_dim=int(obj["output_dim"]),
                           n_experts=int(obj["n_experts"]), top_k=int(obj["top_k"]),
                           gate_hidden=int(obj.get("gate_hidden", 0)), lora_rank=int(obj["lora_r"]),
                           lora_alpha=int(obj["lora_alpha"]), lora_dropout=float(obj["lora_dropout"]),
                           pretrained_weight=obj["pretrained_weight"].to(dev, dt)).to(dev, dt)
    with torch.no_grad():
        m.lora_A.copy_(obj["lora_A"].to(dev, dt)); m.lora_B.copy_(obj["lora_B"].to(dev, dt))
        m.gate.weight.copy_(obj["gate_weight"].to(dev, dt)); m.gate.bias.copy_(obj["gate_bias"].to(dev, dt))
    return m.eval()


def router(m, x):
    probs = F.softmax(m.gate(x), dim=-1)
    top1 = probs.argmax(dim=-1)  # noqa: F841
    _, topi = torch.topk(probs, k=m.top_k, dim=-1)
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topi, 1.0)
    probs = probs * mask
    return probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)


def expert(m, x, probs, base):
    x_d = m.lora_dropout(x)
    Ax = torch.einsum("...h,erh->...er", x_d, m.lora_A)
    BAx = torch.einsum("...er,eor->...eo", Ax, m.lora_B)
    update = torch.einsum("...e,...eo->...o", probs, BAx)
    return base + m.scaling * update


@torch.no_grad()
def bench(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(ITERS):
        fn()
    e.record(); torch.cuda.synchronize()
    return 1e3 * s.elapsed_time(e) / ITERS  # µs / call


def main():
    out = os.path.join(HERE, "results", "v2", "microbench_moe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.manual_seed(0)
    res = {"_meta": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "warmup": WARM, "iters": ITERS,
                     "unit": "us_per_call"}}
    for ds, sp in SETS.items():
        for t in sp["steps"]:
            m = build(ckpt(ds, t))
            for phase, S in (("decode", 1), ("prefill", sp["S"])):
                for B in (1, sp["bs"]):
                    x = torch.randn(B, S, 14336, device="cuda", dtype=torch.bfloat16)
                    with torch.no_grad():
                        probs, base = router(m, x), m.linear(x)
                    r = {"n_experts": int(m.n_experts), "shape": [B, S, 14336],
                         "base_linear": bench(lambda: m.linear(x)),
                         "router": bench(lambda: router(m, x)),
                         "expert": bench(lambda: expert(m, x, probs, base)),
                         "moe_full": bench(lambda: m(x))}
                    r["added_vs_base"] = r["moe_full"] - r["base_linear"]
                    key = f"{ds}_t{t}_{phase}_bs{B}"
                    res[key] = r
                    print(f"[{key}] E={r['n_experts']} linear {r['base_linear']:.1f} router {r['router']:.1f} "
                          f"expert {r['expert']:.1f} moe {r['moe_full']:.1f} (+{r['added_vs_base']:.1f}) us", flush=True)
            del m; torch.cuda.empty_cache()
    json.dump(res, open(out, "w"), indent=1)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
