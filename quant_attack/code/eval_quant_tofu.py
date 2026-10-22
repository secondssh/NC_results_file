#!/usr/bin/env python3
"""TOFU eval under 4-bit quantization — RULE (및 base) 용. eval_quant_clinc.py 의 TOFU 판.

평가 경로는 modify_code/mmain_tofu_fixed.py 의 eval_only 분기 + evaluate_and_save 와 같다:
  load_model -> load_lora_moe_only -> custom_evaluate_tofu(forget, retained, real_authors, world_facts)
다른 점은 MoE 를 붙인 뒤 quant.mode 에 따라 양자화하는 것뿐이다.

  quant.mode : none | backbone | all   (all = backbone + MoE lora_A/lora_B/gate)
  quant.impl : fake (기본) | bnb

사용 예 (t19, backbone 4bit):
  cd modify_code && python ../quant_attack/eval_quant_tofu.py --config-name forget \
      data_name=tofu model_path=./train31Llama/trained_model/O3_LLAMA31_TOFU ... \
      +quant.mode=backbone +quant.moe_ckpt=<.../t19_AG20/lora_moe.pt> +quant.out_file=<...>.json
"""
import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import ListConfig

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
MODIFY_CODE = os.path.join(_REPO, "modify_code")
for p in (MODIFY_CODE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from src_lunar.eval_util_tofu import custom_evaluate_tofu               # noqa: E402
from src_lunar.estimated_net_moe_utils import load_lora_moe_only        # noqa: E402
from src_lunar.estimated_net_utils import _get_down_proj                # noqa: E402

from eval_quant_clinc import _sel, router_stats, set_seed               # noqa: E402
from fake_quant import apply_fake_quant, fake_quantize_adapters_        # noqa: E402
from quant_utils import (                                               # noqa: E402
    describe_quantization,
    load_model_quant,
    quantize_moe_backbone_,
)

os.environ["WANDB_DISABLED"] = "true"


@hydra.main(version_base=None, config_path=os.path.join(MODIFY_CODE, "config"),
            config_name="forget")
def main(cfg):
    set_seed(cfg.seed)
    os.chdir(MODIFY_CODE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_idxs = list(cfg.layer_modified)

    mode = _sel(cfg, "quant.mode", "none")
    impl = _sel(cfg, "quant.impl", "fake")
    if mode not in ("none", "backbone", "all") or impl not in ("fake", "bnb"):
        raise ValueError(f"bad quant.mode={mode} / quant.impl={impl}")
    load_bits = 4 if (impl == "bnb" and mode != "none") else 0
    moe_ckpt = _sel(cfg, "quant.moe_ckpt", None)
    do_router = bool(_sel(cfg, "quant.router_stats", True))
    router_max = int(_sel(cfg, "quant.router_max_samples", 0))
    out_file = _sel(cfg, "quant.out_file", None)
    tag = _sel(cfg, "quant.tag", "run")

    print(f"[QUANT] tag={tag} mode={mode} impl={impl} moe_ckpt={moe_ckpt or '(none)'}")
    print(f"[LOAD] {cfg.model_family} from {cfg.model_path}")
    model_base = load_model_quant(cfg.model_family, cfg.model_path, device, bits=load_bits)

    if moe_ckpt:
        # mmain_tofu_fixed.py eval_only 와 같은 호출 (deepcopy 없이 base 에 바로 끼움)
        target_device = next(
            model_base.model_block_modules[layer_idxs[0]].mlp.up_proj.parameters()
        ).device
        model_base = load_lora_moe_only(
            model=model_base,
            load_path=moe_ckpt,
            layer_idxs=layer_idxs,
            device=target_device,
            dtype=torch.bfloat16,
        )
        moe = _get_down_proj(model_base, int(layer_idxs[0]))
        print(f"[MoE] n_experts={moe.n_experts} top_k={moe.top_k}")

    else:
        # base(소거 전) 모델: layer16 이 MoE 가 아니라 eval_util_tofu.compute_router_accuracy_BD 가
        # nn.Linear 에 return_router_logits 를 넘겨 죽는다. router 칸만 비우고 나머지 평가는 그대로.
        import src_lunar.eval_util_tofu as _eut

        def _no_router(name, model, tokenizer, batch_gpu, topics, layer_idx_list, forget_label_idx=1):
            B = batch_gpu["input_ids"].size(0)
            info = {"router_pred_expert": None, "router_target_expert": None, "router_layer_preds": []}
            return [0.0] * B, [dict(info) for _ in range(B)]
        _eut.compute_router_accuracy_BD = _no_router
        print("[MoE] none (base model) — router accuracy disabled")

    fake_report, moe_quant_report = None, None
    if impl == "fake":
        fake_report = apply_fake_quant(model_base.model, mode)
    elif mode != "none":
        if moe_ckpt:
            moe_quant_report = quantize_moe_backbone_(model_base, layer_idxs, bits=4)
        if mode == "all":
            fake_report = {"adapters": fake_quantize_adapters_(model_base.model)}

    model_base._eval()
    quant_desc = describe_quantization(model_base, layer_idxs)
    print(f"[QUANT] {json.dumps(quant_desc, ensure_ascii=False)}")

    # mmain_tofu_fixed.evaluate_and_save 의 tofu 분기와 같은 이름/타깃/순서
    eval_jobs = [
        dict(name="forget", eval_target="forget_edge", data_path=cfg.sampled_data_path),
        dict(name="retained", eval_target="retained_edge", data_path=cfg.retain_data_path),
        dict(name="real_authors", eval_target="factual_edge", data_path=cfg.utiliy_data_path1),
        dict(name="world_facts", eval_target="factual_edge", data_path=cfg.utiliy_data_path2),
    ]
    wanted = _sel(cfg, "quant.eval_splits", None)
    if wanted:
        keep = [str(s).strip() for s in
                (wanted if isinstance(wanted, (list, ListConfig)) else str(wanted).split(","))]
        unknown = [s for s in keep if s and s not in {j["name"] for j in eval_jobs}]
        if unknown:
            raise ValueError(f"unknown eval_splits {unknown}")
        eval_jobs = [j for j in eval_jobs if j["name"] in keep]
    print(f"[EVAL] running splits: {[j['name'] for j in eval_jobs]}")

    all_eval_logs = {}
    for job in eval_jobs:
        print(f"[EVAL] {job['name']} | target={job['eval_target']} | path={job['data_path']}")
        eval_logs = custom_evaluate_tofu(
            cfg=cfg,
            name=job["name"],
            data_path=job["data_path"],
            tokenizer=model_base.tokenizer,
            model=model_base,
            eval_target=job["eval_target"],
            output_es_score=cfg.compute_es_score,
        )
        all_eval_logs[job["name"]] = {
            "eval_target": job["eval_target"],
            "data_path": job["data_path"],
            "results": eval_logs,
        }

    router = None
    if do_router and moe_ckpt:
        router = {
            sp: router_stats(model_base, path, layer_idxs,
                             batch_size=int(cfg.eval_batch_size), max_samples=router_max)
            for sp, path in (("forget", cfg.sampled_data_path), ("retained", cfg.retain_data_path))
        }
        for k, v in router.items():
            if v:
                print(f"[ROUTER] {k}: forget-expert token rate = "
                      f"{v['mean_forget_expert_token_rate']:.4f}  "
                      f"frac={np.round(v['token_expert_frac'], 3).tolist()}")

    all_eval_logs["_quant_meta"] = {
        "tag": tag,
        "mode": mode,
        "impl": impl,
        "bits": 4 if mode != "none" else 0,
        "fake_quant": fake_report,
        "moe_backbone_quant_rel_error": moe_quant_report,
        "model_path": cfg.model_path,
        "moe_ckpt": moe_ckpt,
        "layer_modified": layer_idxs,
        "task_id": cfg.task_id,
        "quantization": quant_desc,
        "router": router,
    }

    save_file = out_file or os.path.join(cfg.save_path, f"forget_{cfg.task_id}.json")
    os.makedirs(os.path.dirname(save_file) or ".", exist_ok=True)
    print(f"Saving completions to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(all_eval_logs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
