"""4-bit quantization helpers for the quantization attack on unlearned models.

배경: Zhang et al. 의 quantization attack 은 "unlearned 모델을 4/8bit 로 양자화하면
원래(소거 전) 가중치의 양자화 형태에 가까워져서 잊었던 지식이 되살아난다" 는 관찰이다.
GA/NPO 처럼 backbone 가중치를 조금만 움직이는 방법은 반올림 한 번에 원위치로 돌아간다.

RULE 은 망각이 backbone delta 가 아니라 **추가된 LoRA MoE + router** 에 들어있으므로
이 메커니즘이 그대로 적용되지 않는다. 그래서 RULE 은 "backbone 만 4bit, LoRA/gate 는 bf16"
으로 평가한다.

주의 — layer 16 함정:
  load_lora_moe_only 는 layer 16 의 down_proj 를 LUNAR_LoRA_MoE_net 으로 **교체**하는데,
  이 모듈은 내부에 backbone 원본 가중치 사본(`pretrained_weight` -> `self.linear`)을 들고 있다.
  4bit 로 로드한 뒤 MoE 를 붙이면 다른 레이어는 전부 4bit 인데 layer 16 backbone 만
  bf16 로 남는다. 하필 RULE 이 작동하는 레이어라 "거기만 양자화를 면제했다" 는 반론이 가능하다.
  그래서 `quantize_moe_backbone_` 로 self.linear.weight 에도 같은 NF4 를 적용한다.
  LUNAR_LoRA_MoE_net.forward 를 Linear4bit 용으로 고쳐 쓸 필요는 없다 — bnb 도 matmul
  직전에 compute_dtype 으로 dequantize 하므로, NF4 양자화 후 bf16 으로 되돌리는
  fake quantization 이 수치적으로 동등하고 구현이 단순하다.

modify_code/ 는 읽기 전용이다. Llama31Model 을 상속해서 _load_model 만 override 한다.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)                      # ScienceQA_expriment
MODIFY_CODE = os.path.join(_REPO, "modify_code")
if MODIFY_CODE not in sys.path:
    sys.path.insert(0, MODIFY_CODE)

from transformers import AutoModelForCausalLM, BitsAndBytesConfig      # noqa: E402

from src_lunar.model_utils.llama31model import Llama31Model            # noqa: E402
from src_lunar.estimated_net_utils import _get_down_proj               # noqa: E402

# bnb Linear4bit 기본값과 맞춘다 (transformers 가 4bit 레이어를 만들 때 쓰는 값)
NF4_BLOCKSIZE = 64


def make_bnb_config(bits, compute_dtype=torch.bfloat16, double_quant=True):
    """bits=4 -> NF4 BitsAndBytesConfig, bits=8 -> int8, 그 외 -> None."""
    if bits in (None, 0, 16, "16", "none", "bf16"):
        return None
    bits = int(bits)
    if bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    if bits == 8:
        return BitsAndBytesConfig(load_in_8bit=True)
    raise ValueError(f"unsupported bits={bits} (expected 4, 8, or 0/None)")


class QuantLlama31Model(Llama31Model):
    """Llama31Model 과 동일하되 quantization_config 만 추가로 넘긴다.

    Llama31Model._load_model 은 torch_dtype/device_map 이 하드코딩이고
    quantization_config 주입구가 없다. modify_code/ 를 수정하지 않기 위해 상속한다.
    tokenizer / tokenize_instructions_fn / block modules 등 나머지는 전부 그대로다.
    """

    _bnb_config = None          # __init__ 전에 클래스 속성으로 심는다

    def _load_model(self, model_path, dtype=torch.bfloat16):
        kwargs = dict(
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="balanced",
        )
        if self._bnb_config is not None:
            kwargs["quantization_config"] = self._bnb_config
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).eval()
        model.requires_grad_(False)
        return model


def load_model_quant(model_family, model_path, device, bits=0,
                     compute_dtype=torch.bfloat16, double_quant=True):
    """bits=0 이면 src_lunar 의 load_model 과 동일, bits=4/8 이면 양자화 로드.

    양자화된 모델은 `.to(device)` 로 옮길 수 없으므로(bnb 가 이미 배치함)
    bits>0 일 때는 _to() 를 호출하지 않는다.
    """
    if model_family != "llama3.1-8b-instruct":
        raise ValueError(
            f"quant attack currently only wired for llama3.1-8b-instruct, got {model_family}"
        )

    bnb = make_bnb_config(bits, compute_dtype=compute_dtype, double_quant=double_quant)
    if bnb is None:
        from src_lunar.model_utils.model_loader import load_model
        print(f"[QUANT] bits=0 -> bf16 (unquantized) load of {model_path}")
        return load_model(model_family, model_path, device)

    print(f"[QUANT] loading {model_path} in {bits}-bit "
          f"(nf4, double_quant={double_quant}, compute_dtype={compute_dtype})")

    class _Cfgd(QuantLlama31Model):
        pass
    _Cfgd._bnb_config = bnb

    model_base = _Cfgd(model_path)
    # 양자화 모델은 bnb 가 이미 GPU 에 올려둔다. _to() 를 부르면 bnb 가 에러를 낸다.
    return model_base


# --------------------------------------------------------------------------- #
# fake NF4 (quantize -> dequantize) : 커스텀 nn.Linear 가중치용
# --------------------------------------------------------------------------- #
def fake_quantize_nf4(weight, double_quant=True, blocksize=NF4_BLOCKSIZE):
    """weight 를 NF4 로 양자화한 뒤 원래 dtype 으로 되돌린다.

    bnb Linear4bit 이 matmul 직전에 하는 dequantize 와 같은 값을 미리 만들어 둔 것이라
    forward 경로를 바꾸지 않고도 "이 가중치는 4bit 다" 를 재현한다.
    """
    from bitsandbytes.functional import quantize_4bit, dequantize_4bit

    orig_dtype, orig_device = weight.dtype, weight.device
    w = weight.detach()
    if w.device.type != "cuda":
        w = w.cuda()
    # bnb 는 fp16/bf16/fp32 만 받는다
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        w = w.to(torch.bfloat16)

    qw, state = quantize_4bit(
        w, blocksize=blocksize, compress_statistics=double_quant, quant_type="nf4"
    )
    deq = dequantize_4bit(qw, state, blocksize=blocksize, quant_type="nf4")
    return deq.to(device=orig_device, dtype=orig_dtype)


def quantize_moe_backbone_(model, layer_idxs, bits=4, double_quant=True):
    """LUNAR_LoRA_MoE_net.self.linear (= backbone down_proj 사본) 만 NF4 로 만든다.

    lora_A / lora_B / gate 는 건드리지 않는다 — RULE 은 LoRA 모듈이라 양자화 대상이 아니다.
    돌려주는 값은 레이어별 (상대 L2 오차) 딕셔너리로, 양자화가 실제로 걸렸는지 확인용이다.
    """
    if int(bits) != 4:
        raise ValueError(f"quantize_moe_backbone_ only implements 4-bit, got {bits}")

    report = {}
    for l in layer_idxs:
        mod = _get_down_proj(model, int(l))
        if mod.__class__.__name__ != "LUNAR_LoRA_MoE_net":
            print(f"[QUANT] layer {l}: not a LUNAR_LoRA_MoE_net, skip backbone quant")
            continue
        w = mod.linear.weight
        before = w.detach().clone()
        deq = fake_quantize_nf4(w, double_quant=double_quant)
        with torch.no_grad():
            w.copy_(deq)
        rel = (before.float() - deq.float()).norm() / before.float().norm()
        report[int(l)] = float(rel)
        print(f"[QUANT] layer {l}: MoE backbone (self.linear) -> NF4, "
              f"relative L2 error = {rel:.4f}")
    return report


def describe_quantization(model_base, layer_idxs=()):
    """모델이 실제로 어떤 dtype/양자화 상태인지 요약한다 (로그/결과 json 용)."""
    hf = model_base.model
    kinds = {}
    for _, m in hf.named_modules():
        cls = m.__class__.__name__
        if cls in ("Linear4bit", "Linear8bitLt", "Linear", "LUNAR_LoRA_MoE_net"):
            kinds[cls] = kinds.get(cls, 0) + 1
    out = {"module_counts": kinds}
    for l in layer_idxs:
        try:
            mod = _get_down_proj(model_base, int(l))
        except TypeError:
            continue
        info = {"class": mod.__class__.__name__}
        if hasattr(mod, "linear"):
            info["backbone_dtype"] = str(mod.linear.weight.dtype)
            info["n_unique_backbone_values"] = int(
                torch.unique(mod.linear.weight.detach().float().flatten()[:65536]).numel()
            )
        if hasattr(mod, "lora_A"):
            info["lora_dtype"] = str(mod.lora_A.dtype)
            info["n_experts"] = int(mod.n_experts)
        out[f"layer_{l}"] = info
    return out
