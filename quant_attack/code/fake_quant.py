"""Fake NF4 4-bit quantization (quantize -> dequantize, 값은 bf16 으로 둔다).

bnb Linear4bit 도 matmul 직전에 NF4 -> compute_dtype(bf16) 로 dequantize 하므로
가중치를 미리 같은 값으로 바꿔 두면 수치적으로 같은 모델이 된다. 장점:
  - RULE(LUNAR_LoRA_MoE_net) / O3(hacked PEFT LoRA) 어느 구조든 forward 를 안 고친다
  - bf16 matmul 속도 그대로 (bnb 4bit 생성보다 빠름)
  - RULE 과 O3 를 완전히 같은 규칙으로 양자화한다

이 파일은 modify_code / O3 코드 어느 쪽도 import 하지 않는다.
(두 저장소 모두 `src` 패키지가 있어서 sys.path 를 건드리면 서로 가린다.)

양자화 범위 (mode)
  none     : 아무것도 안 함 (원래 결과 재현 확인용)
  backbone : 백본 nn.Linear 전부 (32층 x 7 = 224개, lm_head 제외 = bnb 기본과 동일)
             - RULE: layer16 MoE 안의 백본 사본 `.linear` 포함, `.gate` / lora_A / lora_B 제외
             - O3  : PEFT 가 감싼 `base_layer` 포함, lora_A / lora_B 제외
  all      : backbone + 추가 모듈
             - RULE: lora_A [E,r,H], lora_B, gate 의 모든 Linear weight
             - O3  : 모든 lora_A.* / lora_B.* weight
bias / LayerNorm / embedding 은 bnb 와 마찬가지로 건드리지 않는다.
"""
import torch
import torch.nn as nn

NF4_BLOCKSIZE = 64          # bnb Linear4bit 기본값
N_BACKBONE_LINEAR = 224     # llama-3.1-8B: 32 layers x (q,k,v,o,gate,up,down)
MODES = ("none", "backbone", "all")


@torch.no_grad()
def fake_quantize_nf4(weight, double_quant=True, blocksize=NF4_BLOCKSIZE):
    """weight 를 NF4 로 양자화했다가 원래 dtype/device/shape 로 되돌린 텐서를 반환."""
    from bitsandbytes.functional import dequantize_4bit, quantize_4bit

    orig_dtype, orig_device = weight.dtype, weight.device
    w = weight.detach()
    if w.device.type != "cuda" and torch.cuda.is_available():
        w = w.cuda()
    if w.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        w = w.to(torch.bfloat16)
    w = w.contiguous()
    qw, state = quantize_4bit(w, blocksize=blocksize, compress_statistics=double_quant,
                              quant_type="nf4")
    deq = dequantize_4bit(qw, state, blocksize=blocksize, quant_type="nf4")
    return deq.reshape(weight.shape).to(device=orig_device, dtype=orig_dtype)


@torch.no_grad()
def fake_quantize_int3(weight, blocksize=NF4_BLOCKSIZE):
    """3bit: 블록(64) absmax 기준 대칭 정수 7단계 {-3..3}/3 (NF 계열 3bit 코드북이 bnb 에 없어 단순 대칭 int3)."""
    w = weight.detach().float()
    if w.device.type != "cuda" and torch.cuda.is_available():
        w = w.cuda()
    x = w.reshape(-1, blocksize)
    amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    deq = torch.round(x / amax * 3).clamp(-3, 3) / 3 * amax
    return deq.reshape(weight.shape).to(device=weight.device, dtype=weight.dtype)


NBITS = 4   # apply_fake_quant(nbits=...) 가 바꿈


@torch.no_grad()
def _quantize_param_(p, double_quant=True):
    before = p.detach().float().clone()   # fp32 는 .float() 가 같은 텐서라 clone 필요
    deq = fake_quantize_int3(p) if NBITS == 3 else fake_quantize_nf4(p, double_quant=double_quant)
    p.copy_(deq)
    err = (before - deq.float()).norm().item()
    return err ** 2, before.norm().item() ** 2


def _segments(name):
    return name.split(".")


def _is_adapter(name):
    """RULE gate / lora_A / lora_B 혹은 O3 lora_A / lora_B 에 속한 모듈인지."""
    seg = _segments(name)
    return "lora_A" in seg or "lora_B" in seg or "gate" in seg   # gate_proj 는 'gate' segment 아님


def _is_rule_moe(m):
    return m.__class__.__name__ in ("LUNAR_LoRA_MoE_net", "LUNAR_NN_MoE_net")


@torch.no_grad()
def fake_quantize_backbone_(model, double_quant=True, expect=N_BACKBONE_LINEAR):
    """백본 nn.Linear 전부를 fake NF4 로. 반환: 개수 / 전체 상대 L2 오차 / 이름 일부."""
    names, e2, n2 = [], 0.0, 0.0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        if name.split(".")[-1] == "lm_head" or _is_adapter(name):
            continue
        a, b = _quantize_param_(m.weight, double_quant)
        e2 += a; n2 += b
        names.append(name)
    if expect is not None and len(names) != expect:
        raise RuntimeError(f"[fake_quant] backbone Linear {len(names)}개 (기대 {expect}). "
                           f"모듈 이름 규칙 확인 필요: {names[:3]} ... {names[-3:]}")
    rep = {"n_tensors": len(names), "rel_l2": (e2 / max(n2, 1e-12)) ** 0.5,
           "examples": names[:2] + names[-2:]}
    print(f"[fake_quant] backbone: {rep['n_tensors']} Linear -> NF4, rel L2 err {rep['rel_l2']:.4f}")
    return rep


@torch.no_grad()
def fake_quantize_adapters_(model, double_quant=True):
    """RULE MoE(lora_A/lora_B/gate) 와 O3 LoRA(lora_A/lora_B) 가중치를 fake NF4 로."""
    names, e2, n2 = [], 0.0, 0.0
    for name, m in model.named_modules():
        if _is_rule_moe(m):                          # RULE: Parameter 로 들고 있는 expert 들
            for pn in ("lora_A", "lora_B", "expert_W"):
                p = getattr(m, pn, None)
                if isinstance(p, torch.Tensor):
                    a, b = _quantize_param_(p, double_quant); e2 += a; n2 += b
                    names.append(f"{name}.{pn}")
            for gn, gm in m.gate.named_modules() if isinstance(m.gate, nn.Module) else []:
                if isinstance(gm, nn.Linear):
                    a, b = _quantize_param_(gm.weight, double_quant); e2 += a; n2 += b
                    names.append(f"{name}.gate{('.' + gn) if gn else ''}")
        elif isinstance(m, nn.Linear) and ("lora_A" in _segments(name) or "lora_B" in _segments(name)):
            a, b = _quantize_param_(m.weight, double_quant); e2 += a; n2 += b   # O3 LoRA
            names.append(name)
    if not names:
        raise RuntimeError("[fake_quant] 양자화할 adapter(MoE/LoRA) 가중치를 못 찾음")
    rep = {"n_tensors": len(names), "rel_l2": (e2 / max(n2, 1e-12)) ** 0.5,
           "examples": names[:2] + names[-2:]}
    print(f"[fake_quant] adapters: {rep['n_tensors']} tensors -> NF4, rel L2 err {rep['rel_l2']:.4f}")
    return rep


def apply_fake_quant(model, mode, double_quant=True, expect_backbone=N_BACKBONE_LINEAR, nbits=4):
    """mode in {none, backbone, all}, nbits in {4 (NF4), 3 (대칭 int3)}. 결과 json 에 넣을 report dict 반환."""
    global NBITS
    if mode not in MODES:
        raise ValueError(f"quant mode {mode!r} not in {MODES}")
    if nbits not in (3, 4):
        raise ValueError(f"nbits {nbits} not in (3, 4)")
    NBITS = nbits
    rep = {"mode": mode, "impl": "fake_nf4" if nbits == 4 else "fake_int3", "nbits": nbits,
           "blocksize": NF4_BLOCKSIZE, "double_quant": double_quant}
    if mode == "none":
        return rep
    rep["backbone"] = fake_quantize_backbone_(model, double_quant, expect_backbone)
    if mode == "all":
        rep["adapters"] = fake_quantize_adapters_(model, double_quant)
    return rep
