#!/usr/bin/env python3
"""Targeted relearning attack (Hu et al. 2024) on RULE — CLINC.

공격: 공개된 모델에 공격 데이터로 LoRA(r=8) fine-tuning → 원래 eval 경로(custom_evaluate) 로 재평가.
  attack.target : base     = unlearning 전 모델(MoE 없음)에 같은 공격 (기준선, 데이터가 가르칠 수 있는 상한)
                  backbone = RULE 의 백본 Linear 에만 새 LoRA, MoE(expert/router) 는 배포된 그대로 고정
                  moe      = backbone + RULE MoE 의 lora_A / lora_B / gate 도 학습
                             (top-1 hard routing 이라 gate 는 CE 에서 gradient 가 0 일 수 있음 → 로그에 grad norm 기록)
  attack.data   : 공격 데이터 json (relearn_attack/data/*.json, clincqa train 과 같은 형식)
  attack.epochs : 1 (easy) / 5 (hard),  attack.lr 1e-4, attack.bs 16 (micro 8 × accum 2)
평가는 eval_quant_clinc.py 와 같은 jobs (forget/retained/mrpc/rte), 결과 json 에 _attack_meta 추가.
layer16 down_proj(LUNAR MoE) 는 백본 LoRA 대상에서 제외 (nn.Linear 아님). 학습 후 LoRA 는 merge.

cd modify_code && python ../relearn_attack/relearn_clinc.py --config-name forget <run_quant_clinc.sh 와 같은 인자> \
    +attack.target=backbone +attack.data=<json> +attack.epochs=1 +attack.moe_ckpt=<lora_moe.pt> +attack.out_file=<json>
"""
import json
import math
import os
import random
import sys
import time

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
MODIFY_CODE = os.path.join(_REPO, "modify_code")
for p in (MODIFY_CODE, os.path.join(_REPO, "quant_attack"), _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from peft import LoraConfig, get_peft_model                              # noqa: E402

from src_lunar.eval_util import custom_evaluate, get_dataloader          # noqa: E402
from src_lunar.estimated_net_moe_utils import load_lora_moe_only         # noqa: E402
from src_lunar.estimated_net_utils import _get_down_proj                 # noqa: E402
from src_lunar.model_utils.model_loader import load_model                # noqa: E402

os.environ["WANDB_DISABLED"] = "true"
# layer16 down_proj(LUNAR MoE) 제외한 백본 Linear 전부
TARGET_RE = (r".*\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj))$"
             r"|.*\.layers\.(?!16\.)\d+\.mlp\.down_proj$")


def _sel(cfg, key, default=None):
    v = OmegaConf.select(cfg, key)
    return default if v is None else v


def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def train_attack(cfg, model_base, data_path, epochs, lr, micro_bs, accum, target, seed, impl="peft"):
    tok = model_base.tokenizer
    hf = model_base.model
    hf.config.use_cache = False
    if impl == "custom":     # relearn_attack/attack_lora.py: RULE/GD/O3 공통 구현 (layer16 down_proj 제외 223개)
        from attack_lora import inject
        names = inject(hf)
        pm = hf
    else:
        lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=TARGET_RE, bias="none",
                          task_type="CAUSAL_LM")
        pm = get_peft_model(hf, lcfg)
    moe = None
    if target == "moe":
        moe = _get_down_proj(model_base, 16)
        for n, p in moe.named_parameters():
            if n != "linear.weight":          # 백본 사본은 고정, expert(lora_A/B)·gate 만
                p.requires_grad_(True)
    n_lora = len(names) if impl == "custom" else sum(1 for n, m in pm.named_modules() if n.endswith("lora_A"))
    trainable = [p for p in pm.parameters() if p.requires_grad]
    print(f"[ATTACK] target={target} lora_modules={n_lora} trainable={sum(p.numel() for p in trainable):,}")

    dl0 = get_dataloader(cfg=cfg, data_path=data_path, tokenizer=tok, eval_target="forget_edge")
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(dl0.dataset, batch_size=micro_bs, shuffle=True, collate_fn=dl0.collate_fn, generator=g)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(dl) / accum) * epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0)
    pm.train()
    log, gnorm_gate, step = [], [], 0
    t0 = time.time()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for i, batch in enumerate(dl):
            ids, labels, attn = batch[0], batch[1], batch[2]
            L = int(attn.sum(dim=1).max())                       # right padding → 유효 길이까지만
            ids, labels, attn = ids[:, :L].cuda(), labels[:, :L].cuda(), attn[:, :L].cuda()
            model_base._cur_attn_mask_2d = attn
            out = pm(input_ids=ids, attention_mask=attn, labels=labels)
            (out.loss / accum).backward()
            tot += out.loss.item(); nb += 1
            if (i + 1) % accum == 0 or i + 1 == len(dl):
                if moe is not None and moe.gate.weight.grad is not None:
                    gnorm_gate.append(float(moe.gate.weight.grad.norm()))
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
        log.append(round(tot / max(nb, 1), 4))
        print(f"[ATTACK] epoch {ep + 1}/{epochs} loss {log[-1]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    pm.eval()
    if impl == "custom":        # merge 없이 그대로 (AttackLoRALinear 가 forward 에서 더함)
        pm.config.use_cache = True
    else:
        merged = pm.merge_and_unload()
        merged.config.use_cache = True
        model_base.model = merged
    return dict(epoch_loss=log, steps=step, planned_steps=steps, n_lora_modules=n_lora,
                trainable_params=sum(p.numel() for p in trainable),
                gate_grad_norm_mean=(sum(gnorm_gate) / len(gnorm_gate)) if gnorm_gate else None,
                train_sec=round(time.time() - t0))


@hydra.main(version_base=None, config_path=os.path.join(MODIFY_CODE, "config"), config_name="forget")
def main(cfg):
    os.chdir(MODIFY_CODE)
    seed = int(cfg.seed)
    set_seed(seed)
    target = _sel(cfg, "attack.target")
    assert target in ("base", "backbone", "moe", "none"), target
    moe_ckpt = _sel(cfg, "attack.moe_ckpt")
    data = _sel(cfg, "attack.data")
    epochs = int(_sel(cfg, "attack.epochs", 1))
    lr = float(_sel(cfg, "attack.lr", 1e-4))
    out_file = _sel(cfg, "attack.out_file")
    layer_idxs = list(cfg.layer_modified)

    model_base = load_model(cfg.model_family, cfg.model_path, torch.device("cuda"))
    forget_label_idx = 1
    if target in ("backbone", "moe", "none") and moe_ckpt:
        model_base = load_lora_moe_only(model=model_base, load_path=moe_ckpt, layer_idxs=layer_idxs,
                                        device="cuda", dtype=torch.bfloat16)
        forget_label_idx = int(_get_down_proj(model_base, layer_idxs[0]).n_experts) - 1

    meta = dict(target=target, data=data, epochs=epochs, lr=lr, moe_ckpt=moe_ckpt, seed=seed,
                model_path=cfg.model_path)
    if target != "none":
        impl = _sel(cfg, "attack.impl", "peft")
        meta["impl"] = impl
        meta.update(train_attack(cfg, model_base, data, epochs, lr, micro_bs=8, accum=2,
                                 target=("moe" if target == "moe" else "backbone"), seed=seed, impl=impl))
    model_base._eval()

    jobs = [dict(name="forget", eval_target="forget_edge", data_path=cfg.sampled_data_path),
            dict(name="retained", eval_target="retained_edge", data_path=cfg.retain_data_path),
            dict(name="mrpc", eval_target="factual_edge", data_path=cfg.utiliy_data_path1),
            dict(name="rte", eval_target="factual_edge", data_path=cfg.utiliy_data_path2)]
    res = {}
    for j in jobs:
        print(f"[EVAL] {j['name']}")
        r = custom_evaluate(cfg=cfg, name=j["name"], data_path=j["data_path"], tokenizer=model_base.tokenizer,
                            model=model_base, eval_target=j["eval_target"], output_es_score=cfg.compute_es_score,
                            forget_label_idx=forget_label_idx)
        res[j["name"]] = {"eval_target": j["eval_target"], "data_path": j["data_path"], "results": r}
    res["_attack_meta"] = meta
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    json.dump(res, open(out_file, "w"), indent=2, ensure_ascii=False)
    print(f"[DONE] {out_file} | " + " ".join(f"{k} {v['results']['choice_accuracy']:.3f}"
                                              for k, v in res.items() if k != "_attack_meta"))


if __name__ == "__main__":
    main()
