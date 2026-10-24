#!/usr/bin/env python3
"""expert 수 E 에 따른 RULE 학습 시간 (1 stage). profile_train.py 와 같은 방식으로 원래 학습 스크립트(mmain_tofu_fixed.py, TOFU t19)를
원래 인자로 실행하고, MoE 학습 단계만 E 를 바꿔 1 epoch 씩 잰다 (같은 activation cache, 같은 train_loader).

  모델 로드 / direction 추출 / activation 추출: E 와 무관 (base 모델로 함) → 1 회 측정
  MoE joint 학습: E = 2, 10, 21, 40, 80, 160, 320 각각 1 epoch (원래 학습 = 20 epoch → × 20)
    E < 21 : t19 MoE 의 앞 E 개 expert,  E > 21 : t19 의 21 개 + N(0, 0.01) 랜덤 expert (expand_lora_moe_experts 와 같은 초기화)
    학습 대상 = 마지막 expert (current_forget_idx = E-1, 실제로 E 번째 요청을 학습하는 상황), forget 데이터 label 도 E-1 로 바꿈
    → train_moe.py 가 step 마다 이전 expert 0 … E-2 의 grad 를 하나씩 0 으로 만드는 비용 (zero_expert_param_grads /
       zero_gate_row_grads, expert 수에 비례) 까지 실제와 같게 포함 (cur 를 20 으로 고정하면 E ≥ 40 에서 이 비용이 빠짐)
    optimizer = AdamW(lr = cfg.lr), scheduler = ExponentialLR(0.9) — mmain_tofu_fixed.py 와 같음
  학습용 8B 복사본(deepcopy) 은 제거 (결과 동일 확인됨, profile_train --no_deepcopy 와 같음)
출력: latency/results/v2/timing/scale_train.json
사용 (modify_code 에서): CUDA_VISIBLE_DEVICES=0 python latency/scale_train.py
"""
import argparse
import copy as _copy
import importlib
import json
import os
import sys
import time

import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from profile_train import MC, Rec, overrides  # noqa: E402  (import 시 modify_code 로 chdir)
from src_lunar.estimated_net_moe_utils import LUNAR_LoRA_MoE_net  # noqa: E402

OUT = os.path.join(HERE, "results", "v2", "timing", "scale_train.json")


def resize(src, E, seed=0):
    """src (LUNAR_LoRA_MoE_net) 를 E 개 expert 로: 앞 min(E, n) 개 복사, 나머지 N(0, 0.01)."""
    n = src.n_experts
    dev, dt = src.lora_A.device, src.lora_A.dtype
    m = LUNAR_LoRA_MoE_net(input_dim=src.linear.in_features, output_dim=src.linear.out_features, n_experts=E, top_k=src.top_k,
                           gate_hidden=0, lora_rank=src.lora_r, lora_alpha=src.lora_alpha, lora_dropout=src.lora_dropout.p
                           if hasattr(src.lora_dropout, "p") else 0.0, pretrained_weight=src.linear.weight.data).to(dev, dt)
    g = torch.Generator(device=dev).manual_seed(seed)
    k = min(E, n)
    with torch.no_grad():
        m.lora_A.normal_(0, 0.01, generator=g); m.lora_B.normal_(0, 0.01, generator=g)
        m.gate.weight.normal_(0, 0.01, generator=g); m.gate.bias.zero_()
        m.lora_A[:k].copy_(src.lora_A[:k]); m.lora_B[:k].copy_(src.lora_B[:k])
        m.gate.weight[:k].copy_(src.gate.weight[:k]); m.gate.bias[:k].copy_(src.gate.bias[:k])
    assert abs(m.scaling - src.scaling) < 1e-9
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, nargs="+", default=[2, 10, 21, 40, 80, 160, 320])
    ap.add_argument("--step", type=int, default=19)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    mod = importlib.import_module("mmain_tofu_fixed")
    rec = Rec()
    info = {"ds": "tofu", "t": a.step, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "no_deepcopy": True, "per_E": {}}
    for name in ("load_model", "load_dataset_to_get_direction", "split_raw_dataset_for_forget", "generate_candidate_directions",
                 "load_lora_moe_only", "expand_lora_moe_experts", "prepare_estimated_net_moe_lora_list"):
        if hasattr(mod, name):
            setattr(mod, name, rec.wrap(name, getattr(mod, name)))
    mod.prepare_trainset = rec.wrap("prepare_trainset (activation 추출)", mod.prepare_trainset)

    class _Copy:
        deepcopy = staticmethod(rec.wrap("deepcopy (제거: 같은 모델)", lambda m: m))
    mod.copy = _Copy
    orig_train = mod.train_moe_multiple_layers

    def train(model_list, train_loader, optimizer, scheduler, device, current_forget_idx, num_epochs=20, **kw):
        info.update(full_num_epochs=num_epochs, n_train_batches=len(train_loader), train_batch_size=train_loader.batch_size,
                    n_train_samples=len(train_loader.dataset), real_n_experts=int(model_list[0].n_experts),
                    current_forget_idx=int(current_forget_idx), lr=optimizer.param_groups[0]["lr"])
        src = model_list[0]
        for E in a.E:
            m = resize(src, E)
            cur = E - 1
            fds = train_loader.dataset.datasets[0]          # forget ActivationDataset (label = 원래 current_forget_idx)
            orig_lab = fds.domain_label_list
            fds.domain_label_list = torch.full_like(orig_lab, cur)
            opt = optim.AdamW(m.parameters(), lr=info["lr"])
            sch = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
            rec.wrap(f"moe_train_joint_E{E} (1 epoch)", orig_train)([m], train_loader, opt, sch, device, cur,
                                                                     num_epochs=1, mode="joint", **kw)
            fds.domain_label_list = orig_lab
            p = rec.phases[-1]
            info["per_E"][str(E)] = {"epoch_sec": p["sec"], "peak_allocated_MB": p["peak_allocated_MB"], "current_forget_idx": cur}
            json.dump({**info, "phases": rec.phases}, open(OUT, "w"), indent=1)
            del m, opt, sch; torch.cuda.empty_cache()
        return model_list
    mod.train_moe_multiple_layers = train
    mod.evaluate_and_save = lambda *x, **k: print("[scale_train] 학습 단계 끝 — 평가 건너뜀", flush=True)
    if hasattr(mod, "save_lora_moe_only"):
        mod.save_lora_moe_only = lambda *x, **k: None

    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(MC, "config"), version_base=None):
        cfg = compose(config_name="forget", overrides=overrides("tofu", a.step))
    mod.train(cfg)
    ph = {p["phase"]: p["sec"] for p in rec.phases}
    pre = sum(v for k, v in ph.items() if k.startswith(("load_model", "deepcopy", "generate_candidate", "prepare_trainset", "load_lora", "expand")))
    info["fixed_sec"] = {"load_model+deepcopy": ph.get("load_model", 0) + ph.get("deepcopy (제거: 같은 모델)", 0),
                         "direction": ph.get("generate_candidate_directions", 0), "activation": ph.get("prepare_trainset (activation 추출)", 0),
                         "sum_pre_moe": pre}
    for E, r in info["per_E"].items():
        r["total_stage_sec_est"] = pre + r["epoch_sec"] * info["full_num_epochs"]
    json.dump({**info, "phases": rec.phases}, open(OUT, "w"), indent=1)
    for E, r in info["per_E"].items():
        print(f"[E={E}] epoch {r['epoch_sec']:.1f}s × {info['full_num_epochs']} + 전처리 {pre:.0f}s = {r['total_stage_sec_est']:.0f}s, "
              f"peak {r['peak_allocated_MB']:.0f}MB", flush=True)
    print(f"[done] {OUT}")


if __name__ == "__main__":
    main()
