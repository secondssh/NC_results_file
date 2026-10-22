#!/usr/bin/env python3
"""RULE router 가 섞인 입력(F+R / R+F)을 어느 expert 로 보냈는지 (layer16 gate argmax = top-1 expert).

repro_check/router_token_eval.py 와 같은 방법 (검증됨: 재생성 100% 일치, routing 99.1%+ 일치):
  같은 dataloader 로 만든 prompt 토큰 + 저장된 생성문을 teacher forcing 으로 한 번 forward 하고 gate 직후 중단.
  저장된 router_info.router_pred_expert 는 쓰지 않는다 (left pad + position_ids 없음 → 실제 생성 routing 과 다름).

expert 번호: 0 = retain(원래 모델 경로), k = k 번째 forget 요청의 expert. F 쪽 expert = 저장된 router_target_expert.
샘플별로 기록: prompt 마지막 토큰 expert(첫 생성 토큰을 결정, 논문 router acc 와 같은 위치),
              생성 토큰들의 expert 분포 (F expert / e0 / 그 외 forget expert).

출력: {res_dir}/router_{ds}_t{t}_{order}.json
사용: python multi_attack/router_cross.py --ds clinc --steps 4 8 --orders FR RF --res_dir multi_attack/results
"""
import argparse
import contextlib
import io
import json
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "repro_check"))
import router_token_eval as rte  # noqa: E402  (import 시 modify_code 로 chdir)

from src_lunar.model_utils.model_loader import load_model  # noqa: E402
from src_lunar.estimated_net_moe_utils import load_lora_moe_only  # noqa: E402

CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]


def ckpt(ds, t):
    if ds == "clinc":
        return f"outputs/repro_clinc_lunar_bs64/seed_0/t{t}_{CLINC_DOMAINS[t]}/lora_moe.pt"
    return f"outputs/tofu_ckpt_repro_lunar/seed_0/t{t}_AG{t + 1:02d}/lora_moe.pt"


def classify(e, f_exp):
    return "F" if e == f_exp else ("e0" if e == 0 else "other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["clinc", "tofu"])
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--orders", nargs="+", default=["FR", "RF"])
    ap.add_argument("--res_dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    res_dir = os.path.abspath(a.res_dir)

    todo = [(t, o, os.path.join(res_dir, f"{a.ds}_t{t}_ours_{o}.json"), os.path.join(res_dir, f"router_{a.ds}_t{t}_{o}.json"))
            for t in a.steps for o in a.orders]
    todo = [x for x in todo if os.path.exists(x[2]) and (a.overwrite or not os.path.exists(x[3]))]
    if not todo:
        print("[router] nothing to do"); return

    spec = rte.DS[a.ds]
    util = __import__(spec["util"], fromlist=["x"])
    cfg = rte.make_cfg(spec)
    model = load_model(cfg.model_family, cfg.model_path, torch.device("cuda"))
    model.model.eval()
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    loaded = None
    for t, order, res_path, out_path in todo:
        if loaded != t:  # router_token_eval 과 같이 같은 base 에 step ckpt 를 다시 끼움
            if loaded is not None:
                rec.remove()
            with contextlib.redirect_stdout(io.StringIO()):
                load_lora_moe_only(model=model, load_path=ckpt(a.ds, t), layer_idxs=[rte.LAYER], device="cuda",
                                   dtype=torch.bfloat16)
            moe = model.model_block_modules[rte.LAYER].mlp.down_proj
            n_exp = int(moe.n_experts)
            assert n_exp == t + 2, (n_exp, t)
            rec = rte.GateRecorder(moe.gate)
            loaded = t
        res = json.load(open(res_path))["forget"]
        gt = res["results"]["generated_text"]
        recs, _ = rte.prompt_records(util, cfg, tok, res["data_path"], "forget_edge")
        assert len(recs) == len(gt), (len(recs), len(gt))
        ws = lambda x: re.sub(r"\s", "", util.normalize_text(x))
        assert sum(ws(r[1]) != ws(g[2]) for r, g in zip(recs, gt)) == 0, "gt 순서 불일치"
        g_ids = rte.gen_ids_from_text(tok, [g[1] for g in gt])
        sel = rte.teacher_force(model, rec, [r[0] + g for r, g in zip(recs, g_ids)], a.bs, pad_id)

        samples = []
        for (p, _), g, s in zip(recs, gt, sel):
            f_exp = int(g[4]["router_target_expert"])
            ps, gs = s[:len(p)], s[len(p):]
            cnt = {k: int(sum(classify(int(e), f_exp) == k for e in gs)) for k in ("F", "e0", "other")}
            samples.append({
                "f_expert": f_exp,
                "last": int(ps[-1]), "last_cls": classify(int(ps[-1]), f_exp),
                "gen_majority": rte.majority(gs, n_exp), "n_gen": int(len(gs)), "gen_cnt": cnt,
                "gen_expert_hist": np.bincount(gs, minlength=n_exp).tolist() if len(gs) else [0] * n_exp,
            })
        n = len(samples)
        summ = {f"last_{k}": sum(x["last_cls"] == k for x in samples) / n for k in ("F", "e0", "other")}
        tot = {k: sum(x["gen_cnt"][k] for x in samples) for k in ("F", "e0", "other")}
        T = max(sum(tot.values()), 1)
        summ.update({f"gen_tok_{k}": v / T for k, v in tot.items()})
        json.dump({"ds": a.ds, "t": t, "order": order, "ckpt": ckpt(a.ds, t), "n_experts": n_exp, "n": n,
                   "summary": summ, "samples": samples}, open(out_path, "w"), indent=1)
        print(f"[router] {a.ds} t{t} {order} n={n} " + " ".join(f"{k}={v:.3f}" for k, v in summ.items()), flush=True)
    rec.remove()


if __name__ == "__main__":
    main()
