#!/usr/bin/env python3
"""CLINC eval under 4-bit backbone quantization — GA/NPO baselines and RULE in one script.

평가 경로는 기존과 동일하게 유지한다:
  - eval_jobs / custom_evaluate 호출은 unlearning_baseline_new/eval_clinc.py 및
    modify_code/mmain_fixed.py 의 eval 구간과 같다.
  - 다른 점은 (a) 모델을 4bit 로 올리고 (b) RULE 이면 lora_moe.pt 를 붙인 뒤
    그 안의 backbone 사본도 NF4 로 만드는 것, 둘뿐이다.

이렇게 해야 "양자화 여부" 말고는 열 사이에 차이가 없다. 그래서 비양자화(bf16) 열도
기존 json 을 그대로 쓰지 말고 이 스크립트로 한 번 재서 비교하는 걸 권한다
(+quant.bits=0).

사용 예 (t8, 4bit):
  # GA baseline
  python quant_attack/eval_quant_clinc.py --config-name forget \
      model_family=llama3.1-8b-instruct \
      model_path=<.../clincqa_fullft/ga/seed_0/t8_small_talk> \
      +quant.bits=4 +quant.out_file=<...>.json

  # RULE (ours): backbone 만 4bit, LoRA/gate 는 bf16
  python quant_attack/eval_quant_clinc.py --config-name forget \
      model_path=<.../O3_LLAMA31_ClincQA> \
      +quant.bits=4 +quant.moe_ckpt=<.../t8_small_talk/lora_moe.pt> \
      +quant.out_file=<...>.json
"""
import json
import os
import random
import sys

import hydra
import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
MODIFY_CODE = os.path.join(_REPO, "modify_code")
for p in (MODIFY_CODE, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from src_lunar.dataset_utils import (                                   # noqa: E402
    load_dataset_to_get_direction,
    split_raw_dataset_for_forget,
)
from src_lunar.eval_util import custom_evaluate                         # noqa: E402
from src_lunar.estimated_net_moe_utils import load_lora_moe_only        # noqa: E402
from src_lunar.estimated_net_utils import _get_down_proj                # noqa: E402
from src_lunar.model_utils.llama31model import (                        # noqa: E402
    build_prompt_from_instruction,
)

from quant_utils import (                                               # noqa: E402
    describe_quantization,
    load_model_quant,
    quantize_moe_backbone_,
)
from fake_quant import apply_fake_quant, fake_quantize_adapters_        # noqa: E402

os.environ["WANDB_DISABLED"] = "true"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def _sel(cfg, key, default=None):
    v = OmegaConf.select(cfg, key)
    return default if v is None else v


# --------------------------------------------------------------------------- #
# router 진단: 양자화가 layer-16 hidden state 를 흔들어서 redirection 이 깨지는지
# --------------------------------------------------------------------------- #
@torch.no_grad()
def router_stats(model_base, data_path, layer_idxs, batch_size=16, max_samples=0):
    """forget/retain 입력이 forget expert(!=0) 로 가는 비율을 잰다.

    router 는 tokenize_instructions_llama31_chat (llama3.1 chat template) 으로 만든
    activation 으로 학습됐으므로 여기서도 같은 포맷으로 넣는다.
    task metric 만 보면 'RULE 이 강건하다' 를 주장할 수 없다 — 양자화로 router 가
    forget 입력을 놓치면 redirection 이 안 걸려서 지식이 그대로 돌아오기 때문이다.
    """
    moe = None
    for l in layer_idxs:
        try:
            m = _get_down_proj(model_base, int(l))
        except TypeError:
            continue
        if m.__class__.__name__ == "LUNAR_LoRA_MoE_net":
            moe = m
            break
    if moe is None:
        return None

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    texts = [d["text_input"] for d in raw]
    if max_samples and max_samples > 0:
        texts = texts[:max_samples]

    tokenizer = model_base.tokenizer
    device = next(model_base.model.parameters()).device
    counts = np.zeros(moe.n_experts, dtype=np.int64)
    per_sample = []

    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"      # 뒤쪽 패딩이라야 유효 길이로 자르기 쉽다
    try:
        for i in tqdm(range(0, len(texts), batch_size), desc="router"):
            chunk = texts[i: i + batch_size]
            prompts = [build_prompt_from_instruction(tokenizer, t, system=None) for t in chunk]
            enc = tokenizer(prompts, padding=True, truncation=False,
                            return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            pos = torch.arange(input_ids.size(1), device=device).unsqueeze(0)

            moe.record_expert = True
            moe.expert_history = []
            model_base._cur_attn_mask_2d = attn
            model_base.model(input_ids=input_ids, attention_mask=attn,
                             position_ids=pos, return_dict=True)

            hist = moe.expert_history
            if not hist:
                continue
            sel = hist[-1][..., 0].numpy()            # [B, T]
            lens = attn.sum(dim=1).tolist()
            for b, L in enumerate(lens):
                s = sel[b, :int(L)]
                np.add.at(counts, s, 1)
                per_sample.append(float((s != 0).mean()))
            moe.expert_history = []
            torch.cuda.empty_cache()
    finally:
        moe.record_expert = False
        moe.expert_history = []
        tokenizer.padding_side = prev_side

    total = int(counts.sum())
    return {
        "n_samples": len(per_sample),
        "n_experts": int(moe.n_experts),
        "token_expert_counts": counts.tolist(),
        "token_expert_frac": (counts / max(total, 1)).tolist(),
        "mean_forget_expert_token_rate": float(np.mean(per_sample)) if per_sample else None,
        "frac_samples_majority_forget": float(np.mean(np.array(per_sample) > 0.5))
        if per_sample else None,
    }


@hydra.main(version_base=None, config_path=os.path.join(MODIFY_CODE, "config"),
            config_name="forget")
def main(cfg):
    set_seed(cfg.seed)

    # load_dataset_to_get_direction() 이 상대경로("dataset/splits/harmful.json")를 연다.
    # cfg 로 넘기는 경로는 전부 절대경로라 chdir 해도 안전하다 (eval_clinc.py 와 동일).
    os.chdir(MODIFY_CODE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layer_idxs = list(cfg.layer_modified)

    # quant.mode : none | backbone | all  (all = backbone + MoE lora_A/lora_B/gate)
    # quant.impl : fake (기본, NF4 양자화 후 bf16 복원) | bnb (bitsandbytes 4bit 로드, 예전 방식)
    # 예전 인자 quant.bits 만 주면 bits>0 -> backbone/bnb, 0 -> none 로 해석한다.
    bits = int(_sel(cfg, "quant.bits", 0))
    mode = _sel(cfg, "quant.mode", None)
    impl = _sel(cfg, "quant.impl", None)
    if mode is None:
        mode = "backbone" if bits else "none"
        impl = impl or "bnb"
    impl = impl or "fake"
    if mode not in ("none", "backbone", "all") or impl not in ("fake", "bnb"):
        raise ValueError(f"bad quant.mode={mode} / quant.impl={impl}")
    load_bits = 4 if (impl == "bnb" and mode != "none") else 0
    moe_ckpt = _sel(cfg, "quant.moe_ckpt", None)
    quant_moe_backbone = bool(_sel(cfg, "quant.quantize_moe_backbone", True))
    do_router = bool(_sel(cfg, "quant.router_stats", True))
    router_max = int(_sel(cfg, "quant.router_max_samples", 0))
    out_file = _sel(cfg, "quant.out_file", None)
    tag = _sel(cfg, "quant.tag", "run")

    print(f"[QUANT] tag={tag} mode={mode} impl={impl} moe_ckpt={moe_ckpt or '(none)'} "
          f"quantize_moe_backbone={quant_moe_backbone}")
    print(f"[LOAD] {cfg.model_family} from {cfg.model_path}")
    model_base = load_model_quant(cfg.model_family, cfg.model_path, device, bits=load_bits)

    moe_quant_report = None
    fake_report = None
    forget_label_idx = 1
    if moe_ckpt:
        # mmain_fixed.py 와 같은 방식으로 레이어의 실제 device 를 따라간다 (멀티 GPU 대비)
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
        forget_label_idx = int(moe.n_experts) - 1
        print(f"[MoE] n_experts={moe.n_experts} top_k={moe.top_k} "
              f"forget_label_idx={forget_label_idx}")

    if impl == "fake":
        # MoE 를 붙인 뒤에 양자화해야 layer16 MoE 안 백본 사본(.linear)까지 같이 걸린다
        fake_report = apply_fake_quant(model_base.model, mode, nbits=int(_sel(cfg, "quant.nbits", 4)))
    elif mode != "none":
        if moe_ckpt and quant_moe_backbone:
            # layer 16 만 bf16 로 남는 걸 막는다 (quant_utils 의 'layer 16 함정' 참고)
            moe_quant_report = quantize_moe_backbone_(model_base, layer_idxs, bits=4)
        elif moe_ckpt:
            print("[QUANT] NOTE: MoE backbone left in bf16 — layer "
                  f"{layer_idxs} is NOT quantized (ablation mode)")
        if mode == "all":
            fake_report = {"adapters": fake_quantize_adapters_(model_base.model)}

    model_base._eval()
    quant_desc = describe_quantization(model_base, layer_idxs)
    print(f"[QUANT] {json.dumps(quant_desc, ensure_ascii=False)}")

    # eval_clinc.py / eval_only.py 가 찍는 것과 같은 [DATASET] 카운트 (동작 차이 최소화)
    ref_train, forget_train = load_dataset_to_get_direction(
        cfg, instructions_only=True,
        use_harmful=cfg.use_harmful, use_unverified=cfg.use_unverified,
    )
    print(f"[DATASET] forget_dataset: {len(forget_train)}")
    print(f"[DATASET] ref_dataset: {len(ref_train)}")
    forget_dataset, retain_dataset = split_raw_dataset_for_forget(
        cfg, model_base, instructions_only=True, torch_reformat=False,
    )
    print(f"[DATASET] forget_dataset: {len(forget_dataset)}")
    print(f"[DATASET] retain_dataset: {len(retain_dataset)}")

    # mmain_fixed.py 의 clincqa 분기와 같은 이름/타깃
    eval_jobs = [
        dict(name="forget", eval_target="forget_edge", data_path=cfg.sampled_data_path),
        dict(name="retained", eval_target="retained_edge", data_path=cfg.retain_data_path),
        dict(name="mrpc", eval_target="factual_edge", data_path=cfg.utiliy_data_path1),
        dict(name="rte", eval_target="factual_edge", data_path=cfg.utiliy_data_path2),
    ]
    wanted = _sel(cfg, "quant.eval_splits", None)
    if wanted:
        keep = [str(s).strip() for s in
                (wanted if isinstance(wanted, (list, ListConfig)) else str(wanted).split(","))]
        keep = [s for s in keep if s]
        known = {j["name"] for j in eval_jobs}
        unknown = [s for s in keep if s not in known]
        if unknown:
            raise ValueError(f"unknown eval_splits {unknown}; known: {sorted(known)}")
        eval_jobs = [j for j in eval_jobs if j["name"] in keep]
    print(f"[EVAL] running splits: {[j['name'] for j in eval_jobs]}")

    all_eval_logs = {}
    for job in eval_jobs:
        print(f"[EVAL] {job['name']} | target={job['eval_target']} | path={job['data_path']}")
        eval_logs = custom_evaluate(
            cfg=cfg,
            name=job["name"],
            data_path=job["data_path"],
            tokenizer=model_base.tokenizer,
            model=model_base,
            eval_target=job["eval_target"],
            output_es_score=cfg.compute_es_score,
            forget_label_idx=forget_label_idx,
        )
        all_eval_logs[job["name"]] = {
            "eval_target": job["eval_target"],
            "data_path": job["data_path"],
            "results": eval_logs,
        }

    router = None
    if do_router and moe_ckpt:
        print("[ROUTER] measuring expert selection on forget / retained splits")
        router = {
            "forget": router_stats(model_base, cfg.sampled_data_path, layer_idxs,
                                   batch_size=int(cfg.eval_batch_size),
                                   max_samples=router_max),
            "retained": router_stats(model_base, cfg.retain_data_path, layer_idxs,
                                     batch_size=int(cfg.eval_batch_size),
                                     max_samples=router_max),
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
        "model_path": cfg.model_path,
        "moe_ckpt": moe_ckpt,
        "quantize_moe_backbone": quant_moe_backbone if moe_ckpt else None,
        "moe_backbone_quant_rel_error": moe_quant_report,
        "layer_modified": layer_idxs,
        "task_id": cfg.task_id,
        "quantization": quant_desc,
        "router": router,
    }

    save_file = out_file or os.path.join(cfg.save_path, f"forget_{cfg.task_id}.json")
    save_dir = os.path.dirname(save_file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    print(f"Saving completions to {save_file}")
    with open(save_file, "w", encoding="utf-8") as f:
        json.dump(all_eval_logs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
