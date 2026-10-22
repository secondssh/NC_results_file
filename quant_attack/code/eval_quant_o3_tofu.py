#!/usr/bin/env python3
"""Quantization attack on O3 (baseline) — PTOFU stage_{N} eval-only (seed S).

원래 O3 TOFU 결과(files/results/unlearn_ptofu/llama31_8b_O3/seed_S/stage_N/ptofu.json)는
PTOFU_experiment/src/model/unlearn.py 의 Unlearn.run() 이 stage 마다
  get_ood_weights(stage_N, non_target, real_authors, world_facts) -> eval() -> eval_ptofu()
순서로 만든 것이다. 여기서는 학습 없이 같은 함수들을 그대로 불러 평가만 하고,
PeftModel 로드 직후 fake NF4 (quant_attack/fake_quant.py) 만 끼운다. 원본 코드는 수정하지 않는다.

  --mode none|backbone|all   backbone = 백본 Linear 224개(base_layer), all = + LoRA lora_A/lora_B
  --max_samples N            split 마다 앞 N 개만 (재현 확인용)
OOD detector(RoBERTa) 는 양자화하지 않는다.

실행 (cwd 는 PTOFU_experiment 여야 함: ood_checkpoints_seed_* 가 상대경로):
  cd PTOFU_experiment && python ../ScienceQA_expriment/quant_attack/eval_quant_o3_tofu.py \
      --stage 19 --seed 0 --mode backbone --out_dir <...>
"""
import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))                 # O3-LLM-UNLEARNING
PTOFU = os.path.join(_ROOT, "PTOFU_experiment")
sys.path.insert(0, os.path.join(PTOFU, "src"))
sys.path.insert(1, _HERE)

import torch                                                     # noqa: E402
from transformers import AutoConfig, RobertaTokenizer            # noqa: E402

import metrics.PTofu as ptofu_metrics                            # noqa: E402
from model.unlearn import Unlearn                                # noqa: E402  (rope 패치도 여기서 적용됨)
from model_src.modeling_llama_hacked_o import LlamaForCausalLM_ood   # noqa: E402
from model_src.peft_model_hacked_o import PeftModel              # noqa: E402

from fake_quant import apply_fake_quant                          # noqa: E402

DEFAULT_BASE = os.path.join(_ROOT, "ScienceQA_expriment/modify_code/train31Llama/trained_model/O3_LLAMA31_TOFU")
DEFAULT_RES = os.path.join(PTOFU, "files/results/unlearn_ptofu/llama31_8b_O3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=19)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", default="none", choices=["none", "backbone", "all"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--base_model", default=DEFAULT_BASE)
    ap.add_argument("--o3_root", default=DEFAULT_RES)
    ap.add_argument("--max_samples", type=int, default=0)
    a = ap.parse_args()

    if os.path.abspath(os.getcwd()) != os.path.abspath(PTOFU):
        os.chdir(PTOFU)
    ckpt = os.path.join(a.o3_root, f"seed_{a.seed}", f"stage_{a.stage}", "checkpoints")
    assert os.path.isfile(os.path.join(ckpt, "adapter_model.safetensors")), ckpt
    out_dir = os.path.abspath(a.out_dir)
    assert not out_dir.startswith(os.path.abspath(a.o3_root)), "원래 O3 결과 폴더에 쓰지 않는다"

    if a.max_samples > 0:  # eval_ptofu 내부 _load_records 를 잘라서 split 마다 앞 N 개만
        _orig = ptofu_metrics._load_records
        ptofu_metrics._load_records = lambda name, stage_num=None: _orig(name, stage_num)[:a.max_samples]

    # ---- OOD weights: Unlearn.get_ood_weights 를 그대로 사용 (학습 관련 속성은 불필요) ----
    t0 = time.time()
    u = Unlearn.__new__(Unlearn)
    u.seed, u.model_name = a.seed, a.base_model
    u.device = torch.device("cuda:0")
    u.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "down_proj", "up_proj"]
    ood_tok = RobertaTokenizer.from_pretrained("roberta-large")
    ood_w = u.get_ood_weights(f"stage_{a.stage}", "non_target", ood_tok, a.stage, ood_base_model="roberta-large")

    # ---- 모델: Unlearn.eval() 과 같은 로드 ----
    base_config = AutoConfig.from_pretrained(a.base_model)
    base_config.lora_target_modules = u.lora_target_modules
    base_config.orthogonal_loss = False
    base_config.orthogonal_loss_weight = 0.1
    model = LlamaForCausalLM_ood.from_pretrained(
        a.base_model, config=base_config, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    model.init_olora(orthogonal_loss=False, olora_weights=[])
    model.init_oodweight(ood_weight=[1, 1])
    model.init_active_adapters_d(active_adapters_d=["default"])
    model = PeftModel.from_pretrained(model, ckpt, is_trainable=True)
    model.eval()
    torch.cuda.empty_cache()

    quant_rep = apply_fake_quant(model, a.mode)

    ptofu_metrics.eval_ptofu(
        model=model, model_name=a.base_model, output_dir=out_dir,
        stage_num=a.stage, if_llama=True, ood_weights=ood_w,
    )
    quant_rep.update(stage=a.stage, seed=a.seed, ckpt=ckpt, base_model=a.base_model,
                     max_samples=a.max_samples, elapsed_sec=round(time.time() - t0),
                     ood_weight_mean={k: sum(v) / max(len(v), 1) for k, v in ood_w.items()})
    with open(os.path.join(out_dir, "quant_meta.json"), "w") as f:
        json.dump(quant_rep, f, indent=2)
    print(f"[O3-TOFU quant] mode={a.mode} done in {quant_rep['elapsed_sec']}s -> {out_dir}")


if __name__ == "__main__":
    main()
