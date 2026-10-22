"""방법(RULE / GD / O3)과 무관하게 같은 '공격 LoRA' 를 백본 Linear 에 끼우는 최소 구현 (peft 미사용).

대상: 각 layer 의 q,k,v,o,gate,up,down_proj 의 백본 nn.Linear (O3 는 hacked PEFT 의 `.base_layer`)
      단 layer16 down_proj 는 제외 — RULE 은 그 자리가 MoE 라 공격 LoRA 를 못 붙이므로 세 방법 모두 같은 223개.
LoRA: r=8, alpha=16, A ~ kaiming, B = 0 (peft 기본과 동일), fp32 로 학습.
"""
import math
import re

import torch
import torch.nn as nn

PAT = re.compile(r"layers\.(\d+)\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)(\.base_layer)?$")


class AttackLoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8, alpha=16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        dev = base.weight.device
        self.A = nn.Parameter(torch.empty(r, base.in_features, device=dev, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scale = alpha / r

    @property
    def weight(self):          # 코드 일부가 .weight 로 dtype/device 를 읽음
        return self.base.weight

    def forward(self, x, *args, **kwargs):
        out = self.base(x)
        return out + ((x.float() @ self.A.t()) @ self.B.t() * self.scale).to(out.dtype)


def inject(model, r=8, alpha=16, skip_layer_down=16):
    """대상 nn.Linear 를 AttackLoRALinear 로 교체. 교체한 이름 목록 반환."""
    for p in model.parameters():
        p.requires_grad_(False)
    targets = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Linear) or isinstance(m, AttackLoRALinear):
            continue
        g = PAT.search(name)
        if not g:
            continue
        if int(g.group(1)) == skip_layer_down and g.group(2) == "mlp.down_proj":
            continue
        targets.append(name)
    for name in targets:
        parent, attr = name.rsplit(".", 1)
        pm = model.get_submodule(parent)
        setattr(pm, attr, AttackLoRALinear(getattr(pm, attr), r, alpha))
    return targets


def attack_params(model):
    return [p for m in model.modules() if isinstance(m, AttackLoRALinear) for p in (m.A, m.B)]


def attack_state(model):
    return {n: t.detach().cpu() for n, t in model.state_dict().items() if n.endswith(".A") or n.endswith(".B")}


def load_attack_state(model, sd):
    missing = [k for k in sd if k not in dict(model.named_parameters())]
    assert not missing, f"attack state 키 불일치 {missing[:3]}"
    model.load_state_dict(sd, strict=False)
