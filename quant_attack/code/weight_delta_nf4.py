"""unlearning 변화량(ΔW = W_unlearned − W_base)이 NF4 4bit 양자화 뒤에 얼마나 남는지 (CPU, GPU 불필요).

백본 Linear 가중치마다 bnb NF4 와 같은 규칙(블록 64, 블록 absmax 로 정규화, 16 코드 최근접)으로
base / unlearned 를 각각 양자화해서 비교한다. (double quant 의 absmax 재양자화는 생략 — 영향 미미)
  same_frac   : 두 모델의 양자화 값이 완전히 같은 원소 비율  (높을수록 → 양자화하면 base 로 돌아감)
  keep_ratio  : ||Q(W_u) − Q(W_b)|| / ||W_u − W_b||  (양자화 뒤 남는 변화량 비율, 0 이면 전부 지워짐)
  rel_delta   : ||ΔW|| / ||W_b||,  bin_ratio : median|ΔW| / median(NF4 눈금 간격의 절반)

python quant_attack/weight_delta_nf4.py <base_dir> <unlearned_dir> [out.json]
"""
import json
import os
import sys

import torch
from safetensors import safe_open

NF4 = torch.tensor([-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
                    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
                    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0])
MID = (NF4[1:] + NF4[:-1]) / 2
HALF_GAP = (NF4[1:] - NF4[:-1]) / 2
BLOCK = 64
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def nf4(w):
    """→ (dequantized 값, 원소별 '눈금 간격 절반' 의 절대 크기)"""
    x = w.float().reshape(-1, BLOCK)
    amax = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    idx = torch.bucketize((x / amax).contiguous(), MID)
    deq = NF4[idx] * amax
    gap = torch.minimum(HALF_GAP[(idx - 1).clamp(0, 14)], HALF_GAP[idx.clamp(0, 14)]) * amax
    return deq.reshape(-1), gap.reshape(-1)


def tensor_index(d):
    ip = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(ip):
        idx = json.load(open(ip))["weight_map"]
        return {k: os.path.join(d, v) for k, v in idx.items()}
    single = os.path.join(d, "model.safetensors")          # 샤딩 안 된 저장
    with safe_open(single, "pt") as f:
        return {k: single for k in f.keys()}


def main():
    base_dir, un_dir = sys.argv[1], sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else None
    bi, ui = tensor_index(base_dir), tensor_index(un_dir)
    names = sorted(k for k in bi if k.endswith(".weight") and any(t in k for t in TARGETS))
    assert len(names) == 224, len(names)
    rows, tot = [], dict(same=0, n=0, d2=0.0, q2=0.0, w2=0.0, changed=0)
    for k in names:
        with safe_open(bi[k], "pt") as f:
            wb = f.get_tensor(k)
        with safe_open(ui[k], "pt") as f:
            wu = f.get_tensor(k)
        d = (wu.float() - wb.float()).reshape(-1)
        qb, gap = nf4(wb)
        qu, _ = nf4(wu)
        same = (qb == qu)
        r = dict(name=k, n=d.numel(),
                 changed_frac=float((d != 0).float().mean()),
                 same_frac=float(same.float().mean()),
                 rel_delta=float(d.norm() / wb.float().norm()),
                 keep_ratio=float((qu - qb).norm() / d.norm().clamp_min(1e-20)),
                 bin_ratio=float(d.abs().median() / gap.median()))
        rows.append(r)
        tot["same"] += int(same.sum()); tot["n"] += d.numel(); tot["changed"] += int((d != 0).sum())
        tot["d2"] += float(d.pow(2).sum()); tot["q2"] += float((qu - qb).pow(2).sum())
        tot["w2"] += float(wb.float().pow(2).sum())
    summ = dict(n_tensors=len(rows),
                changed_frac=tot["changed"] / tot["n"],
                same_frac=tot["same"] / tot["n"],
                rel_delta=(tot["d2"] / tot["w2"]) ** 0.5,
                keep_ratio=(tot["q2"] / tot["d2"]) ** 0.5,
                median_bin_ratio=sorted(r["bin_ratio"] for r in rows)[len(rows) // 2])
    print(json.dumps(summ, indent=1))
    by = {}
    for r in rows:
        t = next(t for t in TARGETS if t in r["name"])
        by.setdefault(t, []).append(r)
    for t, rs in by.items():
        print(f"{t:10s} same {sum(x['same_frac'] for x in rs) / len(rs):.4f}  "
              f"keep {sum(x['keep_ratio'] for x in rs) / len(rs):.3f}  "
              f"rel_delta {sum(x['rel_delta'] for x in rs) / len(rs):.2e}  "
              f"bin_ratio {sum(x['bin_ratio'] for x in rs) / len(rs):.3f}")
    if out:
        json.dump(dict(base=base_dir, unlearned=un_dir, summary=summ, per_tensor=rows), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
