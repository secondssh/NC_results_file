#!/usr/bin/env python3
"""섞인 입력에서 F 구간 / R 구간 prompt 토큰이 어느 expert 로 routing 되는지 + 강제 routing 개입 실험 (RULE).

질문: CLINC 는 prompt 마지막 토큰이 e0 로 가도 F 가 안 새는데 TOFU 는 샌다. 왜?
가설: CLINC 는 F 구간 토큰들이 forget expert 로 routing 되어(layer16) 그 표현을 뒤 토큰들이 attention 으로 읽는다.

1) 측정 (--measure, 기본): prompt 만 forward (gate 직후 중단, router_token_eval 과 같은 right pad).
   토큰마다 layer16 top-1 expert. F 구간 / R 구간 (offset mapping 으로 문자 위치 → 토큰) 별로 F expert / e0 / other 비율.
   같은 F 문항 단독 입력(원래 SD 파일)의 질문 구간도 같이 잰다.
2) 개입 (--intervene): "마지막 토큰 e0 & base 는 F 를 답한" 문항에서 prefill 때 F 구간 토큰의 gate 를 강제하고 다시 생성.
     f2e0 : F 구간 → e0      (CLINC: 억제가 F 구간 routing 때문이면 누출이 생겨야 함)
     f2F  : F 구간 → F expert (TOFU: F 구간을 F expert 로 보내면 누출이 줄어야 함)
     none : 개입 없음 (대조군, 저장된 RULE 생성문과 같은지 확인)
   생성은 router_token_eval.validate 와 같은 방식 (left pad, model._generate, greedy, cfg 생성 길이). 채점은 score_cross 와 같은 함수.

출력: {res_dir}/span/{ds}_t{t}_{order}.json (측정), {res_dir}/span/{ds}_t{t}_{order}_int_{mode}.json (개입),
      {res_dir}/span_table.md
사용: CUDA_VISIBLE_DEVICES=1 python multi_attack/router_span.py --ds clinc --steps 4 8 --intervene none f2e0
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "repro_check"))
import router_token_eval as rte  # noqa: E402  (import 시 modify_code 로 chdir)

sys.path.insert(0, HERE)
from router_cross import ckpt, classify  # noqa: E402
from score_cross import side_hits  # noqa: E402

from src_lunar.model_utils.model_loader import load_model  # noqa: E402
from src_lunar.estimated_net_moe_utils import load_lora_moe_only  # noqa: E402


def token_spans(tok, prompt_ids, spans_txt):
    """prompt_ids 를 decode 한 문자열에서 각 문자열(spans_txt)의 마지막 등장 위치를 찾아 토큰 mask 로."""
    text = tok.decode(prompt_ids)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    if enc["input_ids"] != list(prompt_ids):
        return None
    offs = enc["offset_mapping"]
    out = []
    for s, e in spans_txt(text):
        out.append(np.array([a < e and b > s for a, b in offs], bool))
    return out


def cross_char_spans(item):
    q, F, R = item["question"], item["cross"]["F"]["question"], item["cross"]["R"]["question"]
    order = item["cross"]["order"]

    def f(text):
        p = text.rfind(q)
        assert p >= 0
        fs = p if order == "FR" else p + len(q) - len(F)
        rs = p + len(q) - len(R) if order == "FR" else p
        assert text[fs:fs + len(F)] == F and text[rs:rs + len(R)] == R
        return [(fs, fs + len(F)), (rs, rs + len(R))]
    return f


def single_char_span(question):
    def f(text):
        p = text.rfind(question)
        assert p >= 0
        return [(p, p + len(question))]
    return f


def frac(sel, mask, f_exp):
    s = sel[mask]
    if len(s) == 0:
        return {"F": None, "e0": None, "other": None, "n": 0}
    c = [classify(int(e), f_exp) for e in s]
    return {k: c.count(k) / len(c) for k in ("F", "e0", "other")} | {"n": int(len(s))}


class GateForce:
    """prefill(S>1) 호출에서만 mask 위치의 gate logits 를 target expert 로 강제."""

    def __init__(self, gate):
        self.mask, self.target = None, None
        self.h = gate.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        if self.mask is None or out.dim() != 3 or out.size(1) == 1:
            return None
        out = out.clone()
        m = self.mask.to(out.device)
        big = out.detach().abs().max() + 100.0
        tgt = self.target.to(out.device)
        onehot = torch.nn.functional.one_hot(tgt, out.size(-1)).bool()          # [B, E]
        forced = torch.where(onehot[:, None, :], big, -big).to(out.dtype)       # [B, 1, E]
        return torch.where(m[..., None], forced.expand_as(out), out)

    def remove(self):
        self.h.remove()


@torch.no_grad()
def generate(model, cfg, tok, util, prompts, masks, targets, force, bs, pad_id):
    outs = []
    for k in range(0, len(prompts), bs):
        P = prompts[k:k + bs]
        S = max(map(len, P))
        gi = torch.tensor([[pad_id] * (S - len(p)) + p for p in P]).cuda()
        ga = torch.tensor([[0] * (S - len(p)) + [1] * len(p) for p in P]).cuda()
        if masks is not None:
            force.mask = torch.tensor(np.stack([np.r_[np.zeros(S - len(p), bool), m]
                                                for p, m in zip(P, masks[k:k + bs])]))
            force.target = torch.tensor(targets[k:k + bs])
        else:
            force.mask = None
        out = model._generate(input_ids=gi, attention_mask=ga, max_length=cfg.eval_generation_max_length,
                              max_new_tokens=cfg.eval_generation_max_new_tokens, do_sample=False, num_beams=1,
                              num_return_sequences=1, use_cache=True, pad_token_id=pad_id,
                              output_scores=False, return_dict_in_generate=True)
        for row in out.sequences[:, S:].tolist():
            outs.append(util.normalize_text(tok.decode(row, skip_special_tokens=True)))
    force.mask = None
    return outs


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["clinc", "tofu"])
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--orders", nargs="+", default=["FR", "RF"])
    ap.add_argument("--res_dir", default=os.path.join(HERE, "results"))
    ap.add_argument("--intervene", nargs="*", default=[], choices=["none", "f2e0", "f2F"])
    ap.add_argument("--max_items", type=int, default=0, help="개입 문항 수 상한 (0 = 전부)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--gen_bs", type=int, default=0, help="개입 생성 batch (0 = 원래 eval batch)")
    a = ap.parse_args()
    res_dir = os.path.abspath(a.res_dir)
    od = os.path.join(res_dir, "span"); os.makedirs(od, exist_ok=True)
    manifest = json.load(open(os.path.join(HERE, "data", "manifest.json")))

    spec = rte.DS[a.ds]
    util = __import__(spec["util"], fromlist=["x"])
    cfg = rte.make_cfg(spec)
    gen_bs = a.gen_bs or int(cfg.eval_batch_size)
    model = load_model(cfg.model_family, cfg.model_path, torch.device("cuda"))
    model.model.eval()
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    for t in a.steps:
        with contextlib.redirect_stdout(io.StringIO()):
            load_lora_moe_only(model=model, load_path=ckpt(a.ds, t), layer_idxs=[rte.LAYER], device="cuda",
                               dtype=torch.bfloat16)
        moe = model.model_block_modules[rte.LAYER].mlp.down_proj
        assert int(moe.n_experts) == t + 2
        rec = rte.GateRecorder(moe.gate)

        # 단독 F 문항 (원래 SD 파일) 질문 구간 routing
        f_src = manifest["sets"][f"{a.ds}_t{t}"]["F_src"]
        f_data = json.load(open(f_src))
        s_recs, _ = rte.prompt_records(util, cfg, tok, f_src, "forget_edge")
        assert len(s_recs) == len(f_data)

        for order in a.orders:
            data_path = os.path.join(HERE, "data", f"{a.ds}_t{t}_{order}.json")
            items = json.load(open(data_path))
            recs, _ = rte.prompt_records(util, cfg, tok, data_path, "forget_edge")
            assert len(recs) == len(items)
            prompts = [r[0] for r in recs]
            sel = rte.teacher_force(model, rec, prompts, a.bs, pad_id)
            f_idx = sorted({it["cross"]["F"]["src_idx"] for it in items})
            s_sel = dict(zip(f_idx, rte.teacher_force(model, rec, [s_recs[i][0] for i in f_idx], a.bs, pad_id)))

            ours = load_jsonl(os.path.join(res_dir, "samples", f"{a.ds}_t{t}_ours_{order}.jsonl"))
            base = load_jsonl(os.path.join(res_dir, "samples", f"{a.ds}_t{t}_base_{order}.jsonl"))
            rows, fmasks, bad = [], [], 0
            for it, p, s, o, b in zip(items, prompts, sel, ours, base):
                f_exp = int(o["f_expert"])
                ms = token_spans(tok, p, cross_char_spans(it))
                fi = it["cross"]["F"]["src_idx"]
                ss = token_spans(tok, s_recs[fi][0], single_char_span(f_data[fi]["question"]))
                if ms is None or ss is None:
                    bad += 1; fmasks.append(None); rows.append(None); continue
                fmasks.append(ms[0])
                rows.append({"pair_id": it["cross"]["pair_id"], "f_expert": f_exp, "last_cls": o["router_last_cls"],
                             "rule_F_leak": o["F_leak"], "rule_R_ok": o["R_ok"],
                             "base_F_leak": b["F_leak"], "base_R_ok": b["R_ok"],
                             "F_span": frac(s, ms[0], f_exp), "R_span": frac(s, ms[1], f_exp),
                             "single_F_span": frac(s_sel[fi], ss[0], f_exp)})
            json.dump({"ds": a.ds, "t": t, "order": order, "n_bad_offsets": bad, "rows": rows},
                      open(os.path.join(od, f"{a.ds}_t{t}_{order}.json"), "w"), indent=1)
            print(f"[span] {a.ds} t{t} {order}: n={len(rows)} offset 실패 {bad}", flush=True)

            # 개입: 마지막 토큰 e0 & base 가 F 를 답한 문항
            cand = [i for i, r in enumerate(rows) if r and r["last_cls"] == "e0" and r["base_F_leak"]]
            if a.max_items:
                cand = cand[:a.max_items]
            if not a.intervene or not cand:
                continue
            rec.remove()
            force = GateForce(moe.gate)
            for mode in a.intervene:
                masks = None if mode == "none" else [fmasks[i] for i in cand]
                tg = [0 if mode == "f2e0" else rows[i]["f_expert"] for i in cand]
                gens = generate(model, cfg, tok, util, [prompts[i] for i in cand], masks, tg, force, gen_bs, pad_id)
                F_hit = side_hits(a.ds, gens, [items[i]["cross"]["F"] for i in cand], "F")
                R_hit = side_hits(a.ds, gens, [items[i]["cross"]["R"] for i in cand], "R")
                saved_eq = [g == ours[i]["gen"] for g, i in zip(gens, cand)]
                res = {"mode": mode, "n": len(cand), "F_leak": float(np.mean(F_hit)), "R_ok": float(np.mean(R_hit)),
                       "orig_rule_F_leak": float(np.mean([rows[i]["rule_F_leak"] for i in cand])),
                       "orig_rule_R_ok": float(np.mean([rows[i]["rule_R_ok"] for i in cand])),
                       "gen_eq_saved_rule": float(np.mean(saved_eq)),
                       "samples": [{"pair_id": rows[i]["pair_id"], "gen": g, "F_leak": bool(f), "R_ok": bool(r)}
                                   for i, g, f, r in zip(cand, gens, F_hit, R_hit)]}
                json.dump(res, open(os.path.join(od, f"{a.ds}_t{t}_{order}_int_{mode}.json"), "w"), indent=1)
                print(f"[intervene] {a.ds} t{t} {order} {mode}: n={len(cand)} F_leak {res['F_leak']:.3f} "
                      f"(orig {res['orig_rule_F_leak']:.3f}) R_ok {res['R_ok']:.3f} gen==saved {res['gen_eq_saved_rule']:.3f}",
                      flush=True)
            force.remove()
            rec = rte.GateRecorder(moe.gate)
        rec.remove()


if __name__ == "__main__":
    main()
