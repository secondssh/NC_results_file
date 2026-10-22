"""
Token-level router 지표 (prompt token / prompt last token / generated token) + forget/retain confusion matrix.

방법 (teacher forcing, 재생성 없음):
  각 step ckpt 를 base 에 끼우고, 원래 eval 과 같은 dataloader 로 만든 prompt 토큰 뒤에
  결과 json 에 저장된 생성문(generated_text[i][1])을 토크나이즈해 붙여 한 번 forward.
  layer16 MoE gate logits argmax (= top_k=1 선택 expert) 를 토큰마다 기록하고 gate 직후 forward 중단.
  right padding 이라 position 은 실제 생성 때(left pad + position_ids=cumsum(mask)-1)와 같다.
  target 은 결과 json 의 router_info.router_target_expert (retained=0, forget=domain expert).

검증 (--validate K): 첫 step 의 forget/retained 앞 K 배치를 원래 eval 처럼 실제 generate 하면서
  매 forward 의 gate 선택을 기록 → (a) 재생성문 == 저장 생성문, (b) generation 중 routing == teacher forcing routing 비교.

예)
  CUDA_VISIBLE_DEVICES=1 python repro_check/router_token_eval.py --ds clinc --seeds 0 --validate 2 --steps 0
  CUDA_VISIBLE_DEVICES=1 python repro_check/router_token_eval.py --ds tofu --seeds 0 1 2
출력: repro_check/router_metrics/token/{ds}_s{seed}/t{t}_{dom}.json (+ .npz: 샘플별 expert 시퀀스)
"""
import argparse
import contextlib
import glob
import io
import json
import os
import re
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from omegaconf import OmegaConf  # noqa: E402

from src_lunar.model_utils.model_loader import load_model  # noqa: E402
from src_lunar.estimated_net_moe_utils import load_lora_moe_only  # noqa: E402

RES = "run_results/eval_basemoe_error/llama3.1-8b-instruct"
OUT = "repro_check/router_metrics/token"
DS = {
    "tofu": dict(
        run=RES + "/repro_tofu_lunar/tofu_moe_fixed_layer16_{s}",
        ckpt="outputs/tofu_ckpt_repro_lunar/seed_{s}/t{t}_{dom}/lora_moe.pt",
        model_path="./train31Llama/trained_model/O3_LLAMA31_TOFU",
        data_name="tofu", eval_bs=8, util="src_lunar.eval_util_tofu",
    ),
    "clinc": dict(
        run=RES + "/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_{s}",
        ckpt="outputs/repro_clinc_lunar_bs64/seed_{s}/t{t}_{dom}/lora_moe.pt",
        model_path="./train31Llama/trained_model/O3_LLAMA31_ClincQA",
        data_name="clincqa", eval_bs=16, util="src_lunar.eval_util",
    ),
}
SPLITS = [("forget", "forget_edge"), ("retained", "retained_edge")]
# --util: utility split (target = e0). 결과는 t{t}_{dom}_util.json 로 따로 저장 (forget/retained 와 mixed 없음)
UTIL_SPLITS = {"tofu": [("real_authors", "factual_edge"), ("world_facts", "factual_edge")],
               "clinc": [("mrpc", "factual_edge"), ("rte", "factual_edge")]}
LAYER = 16


class _Stop(Exception):
    pass


class GateRecorder:
    """layer16 MoE gate 의 forward hook. stop=True 면 gate 직후 forward 중단."""

    def __init__(self, gate):
        self.calls, self.stop = [], False
        self.h = gate.register_forward_hook(self._hook)

    def _hook(self, mod, inp, out):
        self.calls.append(out.detach().argmax(dim=-1).to("cpu", torch.int16))
        if self.stop:
            raise _Stop

    def remove(self):
        self.h.remove()


def step_dirs(run_dir):
    out = []
    for d in glob.glob(os.path.join(run_dir, "lunar_*_t*_*")):
        m = re.search(r"_t(\d+)_(.+)$", d)
        out.append((int(m.group(1)), m.group(2), d))
    return sorted(out)


def ckpt_path(spec, s, t, dom, sdir):
    ws = os.path.join(sdir, "weights_source.yaml")
    if os.path.exists(ws):
        return json.load(open(ws))["evaluated_weights"]["path"]
    return spec["ckpt"].format(s=s, t=t, dom=dom)


def make_cfg(spec):
    cfg = OmegaConf.load("config/forget.yaml")
    cfg.data_name = spec["data_name"]
    cfg.model_family = "llama3.1-8b-instruct"
    cfg.model_path = spec["model_path"]
    cfg.eval_batch_size = spec["eval_bs"]
    return cfg


def prompt_records(util, cfg, tok, data_path, eval_target):
    """원래 eval 과 같은 dataloader 순서로 (prompt_ids, gt_text) 반환."""
    with contextlib.redirect_stdout(io.StringIO()):  # get_dataloader 가 데이터 전체를 print 함
        dl = util.get_dataloader(cfg=cfg, data_path=data_path, tokenizer=tok, eval_target=eval_target)
    recs = []
    for batch in dl:
        input_ids, labels, attn = batch[0], batch[1], batch[2]
        for i in range(input_ids.size(0)):
            n = int(attn[i].sum())
            ids, lab = input_ids[i, :n], labels[i, :n]
            p = ids[lab.eq(-100)]
            gt = util.normalize_text(tok.decode(ids[lab.ne(-100)].tolist(), skip_special_tokens=True))
            recs.append((p.tolist(), gt))
    return recs, dl


@torch.no_grad()
def teacher_force(model, rec, seqs, bs, pad_id):
    """seqs: list of id lists → list of int16 arrays (토큰별 선택 expert)."""
    order = np.argsort([-len(x) for x in seqs])
    out = [None] * len(seqs)
    dev = model.model.device
    rec.stop = True
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        L = len(seqs[idx[0]])
        ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
        am = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            ids[j, :len(seqs[i])] = torch.tensor(seqs[i])
            am[j, :len(seqs[i])] = 1
        rec.calls.clear()
        try:
            model.model.model(input_ids=ids.to(dev), attention_mask=am.to(dev), use_cache=False)
        except _Stop:
            pass
        sel = rec.calls[-1].numpy()
        for j, i in enumerate(idx):
            out[i] = sel[j, :len(seqs[i])].copy()
    rec.stop = False
    return out


def majority(a, n):
    return int(np.bincount(a, minlength=n).argmax()) if len(a) else -1


def common_affix(seqs):
    """모든 prompt 가 공유하는 앞/뒤 토큰 수 (chat template + instruction 문구)."""
    pre = 0
    while all(len(x) > pre for x in seqs) and len({x[pre] for x in seqs}) == 1:
        pre += 1
    suf = 0
    while all(len(x) > pre + suf for x in seqs) and len({x[-1 - suf] for x in seqs}) == 1:
        suf += 1
    return pre, suf


def metrics(targets, prompt_sel, gen_sel, saved_pred, n_exp, content_sel):
    T = np.asarray(targets)
    r = {"n": int(len(T))}
    c_ok = sum(int((c == t).sum()) for c, t in zip(content_sel, T))
    c_n = sum(len(c) for c in content_sel)
    r["prompt_content_tok_acc"] = c_ok / max(c_n, 1)
    r["prompt_content_majority_acc"] = float(np.mean([majority(c, n_exp) == t for c, t in zip(content_sel, T)]))
    r["n_prompt_content_tok"] = c_n
    cms = {k: np.zeros((n_exp, n_exp), np.int64) for k in
           ["prompt_tok", "prompt_content_tok", "prompt_maj", "last", "gen_tok", "gen_maj"]}
    for c, t in zip(content_sel, T):
        np.add.at(cms["prompt_content_tok"][t], c, 1)
    pt_ok = pt_n = gt_ok = gt_n = 0
    p_mean, g_mean, last_ok, last_eq_saved, gen_empty = [], [], 0, 0, 0
    p_maj_ok = g_maj_ok = 0
    for t, ps, gs, sp in zip(T, prompt_sel, gen_sel, saved_pred):
        np.add.at(cms["prompt_tok"][t], ps, 1)
        pt_ok += int((ps == t).sum()); pt_n += len(ps)
        p_mean.append(float((ps == t).mean()))
        pm = majority(ps, n_exp); cms["prompt_maj"][t, pm] += 1; p_maj_ok += int(pm == t)
        cms["last"][t, ps[-1]] += 1; last_ok += int(ps[-1] == t); last_eq_saved += int(ps[-1] == sp)
        if len(gs) == 0:
            gen_empty += 1
            continue
        np.add.at(cms["gen_tok"][t], gs, 1)
        gt_ok += int((gs == t).sum()); gt_n += len(gs)
        g_mean.append(float((gs == t).mean()))
        gm = majority(gs, n_exp); cms["gen_maj"][t, gm] += 1; g_maj_ok += int(gm == t)
    n_g = len(T) - gen_empty
    r.update({
        "prompt_tok_acc": pt_ok / max(pt_n, 1),            # 토큰 micro
        "prompt_tok_acc_sample_mean": float(np.mean(p_mean)),
        "prompt_majority_acc": p_maj_ok / len(T),
        "prompt_last_acc": last_ok / len(T),
        "prompt_last_eq_saved": last_eq_saved / len(T),     # 저장된 router_info 와 일치율 (검증용)
        "gen_tok_acc": gt_ok / max(gt_n, 1),
        "gen_tok_acc_sample_mean": float(np.mean(g_mean)) if g_mean else None,
        "gen_majority_acc": g_maj_ok / max(n_g, 1),
        "n_gen_empty": gen_empty, "n_prompt_tok": pt_n, "n_gen_tok": gt_n,
    })
    return r, cms


def gen_ids_from_text(tok, texts):
    return [tok.encode(x, add_special_tokens=False) if x else [] for x in texts]


@torch.no_grad()
def validate(util, cfg, model, rec, tok, dl, saved, targets, eos_ids, k_batches, pad_id):
    """원래 run_generation 과 같은 방식으로 실제 생성하며 gate 선택 기록 → teacher forcing 과 비교."""
    n_txt = n_retok = n_tok_eq = n_tok = n_prompt_eq = n_prompt = n_s = 0
    ok_gen = ok_retok = n_retok_tok = 0
    off = 0
    for bi, batch in enumerate(dl):
        if bi >= k_batches:
            break
        input_ids, labels, attn = batch[0], batch[1], batch[2]
        B = input_ids.size(0)
        prompts = []
        for i in range(B):
            n = int(attn[i].sum())
            prompts.append(input_ids[i, :n][labels[i, :n].eq(-100)].tolist())
        S = max(map(len, prompts))
        gi = torch.tensor([[pad_id] * (S - len(p)) + p for p in prompts]).cuda()
        ga = torch.tensor([[0] * (S - len(p)) + [1] * len(p) for p in prompts]).cuda()
        rec.calls.clear()
        out = model._generate(input_ids=gi, attention_mask=ga, max_length=cfg.eval_generation_max_length,
                              max_new_tokens=cfg.eval_generation_max_new_tokens, do_sample=False, num_beams=1,
                              num_return_sequences=1, use_cache=True, pad_token_id=pad_id,
                              output_scores=False, return_dict_in_generate=True)
        calls = [c.numpy() for c in rec.calls]
        new = out.sequences[:, S:].tolist()
        for b in range(B):
            m = next((j for j, x in enumerate(new[b]) if x in eos_ids), len(new[b]))
            g_ids = new[b][:m]
            txt = util.normalize_text(tok.decode(new[b], skip_special_tokens=True))
            n_txt += int(txt == saved[off + b])
            n_retok += int(tok.encode(saved[off + b], add_special_tokens=False) == g_ids)
            p_gen = calls[0][b, S - len(prompts[b]):]
            g_gen = np.array([calls[1 + j][b, 0] for j in range(m)], dtype=np.int16)
            tf = teacher_force(model, rec, [prompts[b] + g_ids], 1, pad_id)[0]
            n_prompt_eq += int((tf[:len(prompts[b])] == p_gen).sum()); n_prompt += len(prompts[b])
            n_tok_eq += int((tf[len(prompts[b]):] == g_gen).sum()); n_tok += m
            # 실제 run 에서 쓰는 방식: 저장 생성문을 재토크나이즈해서 teacher forcing
            rt = tok.encode(saved[off + b], add_special_tokens=False) if saved[off + b] else []
            tr = teacher_force(model, rec, [prompts[b] + rt], 1, pad_id)[0][len(prompts[b]):]
            tg = targets[off + b]
            ok_gen += int((g_gen == tg).sum()); ok_retok += int((tr == tg).sum()); n_retok_tok += len(rt)
            n_s += 1
        off += B
    return dict(samples=n_s, regen_text_eq_saved=n_txt / n_s, prompt_route_eq=n_prompt_eq / max(n_prompt, 1),
                gen_route_eq=n_tok_eq / max(n_tok, 1), n_gen_tok=n_tok, retok_eq=n_retok / n_s,
                gen_acc_generation_time=ok_gen / max(n_tok, 1), gen_acc_tf_retok=ok_retok / max(n_retok_tok, 1),
                n_retok_tok=n_retok_tok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=list(DS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, nargs="*", default=None)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--validate", type=int, default=0, help="첫 step 에서 실제 생성 비교할 배치 수 (0=안 함)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--util", action="store_true", help="forget/retained 대신 utility split 만")
    a = ap.parse_args()
    splits = UTIL_SPLITS[a.ds] if a.util else SPLITS

    spec = DS[a.ds]
    util = __import__(spec["util"], fromlist=["x"])
    cfg = make_cfg(spec)
    model = load_model(cfg.model_family, cfg.model_path, torch.device("cuda"))
    model.model.eval()
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    gc = model.model.generation_config.eos_token_id
    eos_ids = set(gc if isinstance(gc, list) else [gc]) | {tok.eos_token_id}

    validated = False
    for s in a.seeds:
        od = os.path.join(OUT, f"{a.ds}_s{s}")
        os.makedirs(od, exist_ok=True)
        for t, dom, sdir in step_dirs(spec["run"].format(s=s)):
            if a.steps is not None and t not in a.steps:
                continue
            of = os.path.join(od, f"t{t}_{dom}{'_util' if a.util else ''}.json")
            if os.path.exists(of) and not a.overwrite and not a.validate:
                print(f"[skip] {of}"); continue
            t0 = time.time()
            ck = ckpt_path(spec, s, t, dom, sdir)
            with contextlib.redirect_stdout(io.StringIO()):
                load_lora_moe_only(model=model, load_path=ck, layer_idxs=[LAYER], device="cuda", dtype=torch.bfloat16)
            moe = model.model_block_modules[LAYER].mlp.down_proj
            n_exp = int(moe.n_experts)
            assert n_exp == t + 2, (n_exp, t)
            rec = GateRecorder(moe.gate)
            res = json.load(open(glob.glob(os.path.join(sdir, "forget_*.json"))[0]))
            step_out = {"seed": s, "t": t, "domain": dom, "ckpt": ck, "n_experts": n_exp, "splits": {}}
            all_T, all_P, all_G, all_S, all_C = [], [], [], [], []
            npz = {}
            for sp, target in splits:
                gt = res[sp]["results"]["generated_text"]
                recs, dl = prompt_records(util, cfg, tok, res[sp]["data_path"], target)
                assert len(recs) == len(gt), (sp, len(recs), len(gt))
                ws = lambda x: re.sub(r"\s", "", util.normalize_text(x))  # decode 공백 차이 무시
                gt_mis = sum(ws(r[1]) != ws(g[2]) for r, g in zip(recs, gt))
                assert gt_mis == 0, f"{sp}: gt 순서 불일치 {gt_mis}"
                if a.validate and not validated:
                    v = validate(util, cfg, model, rec, tok, dl, [g[1] for g in gt],
                                 [g[4]["router_target_expert"] for g in gt], eos_ids, a.validate, pad_id)
                    print(f"[validate] s{s} t{t} {sp}: {v}")
                    step_out.setdefault("validate", {})[sp] = v
                g_ids = gen_ids_from_text(tok, [g[1] for g in gt])
                sel = teacher_force(model, rec, [r[0] + g for r, g in zip(recs, g_ids)], a.bs, pad_id)
                P = [x[:len(r[0])] for x, r in zip(sel, recs)]
                G = [x[len(r[0]):] for x, r in zip(sel, recs)]
                T = [g[4]["router_target_expert"] for g in gt]
                SP = [g[4]["router_pred_expert"] for g in gt]
                pre, suf = common_affix([r[0] for r in recs])
                C = [x[pre:len(x) - suf] for x in P]
                m, cms = metrics(T, P, G, SP, n_exp, C)
                m["template_prefix_len"], m["template_suffix_len"] = pre, suf
                m["saved_router_accuracy"] = res[sp]["results"]["router_accuracy"]
                step_out["splits"][sp] = {**m, "cm": {k: v.tolist() for k, v in cms.items()}}
                all_T += T; all_P += P; all_G += G; all_S += SP; all_C += C
                npz[f"{sp}_prompt_ids"] = np.concatenate([np.array(r[0], np.int32) for r in recs])
                npz[f"{sp}_target"] = np.array(T, np.int16)
                npz[f"{sp}_prompt_len"] = np.array([len(x) for x in P])
                npz[f"{sp}_gen_len"] = np.array([len(x) for x in G])
                npz[f"{sp}_prompt_sel"] = np.concatenate(P)
                npz[f"{sp}_gen_sel"] = np.concatenate(G) if sum(map(len, G)) else np.zeros(0, np.int16)
            validated = validated or bool(a.validate)
            rec.remove()
            if a.util:
                json.dump(step_out, open(of, "w"), indent=1)
                np.savez_compressed(of[:-5] + ".npz", **npz)
                print(f"[{a.ds} s{s} t{t} util] {time.time() - t0:.0f}s | " + " | ".join(
                    f"{sp} last {v['prompt_last_acc']:.3f} (saved {v['saved_router_accuracy']:.3f}) "
                    f"Pc {v['prompt_content_tok_acc']:.3f} G {v['gen_tok_acc']:.3f}"
                    for sp, v in step_out["splits"].items()), flush=True)
                continue
            m, cms = metrics(all_T, all_P, all_G, all_S, n_exp, all_C)
            step_out["splits"]["mixed"] = {**m, "cm": {k: v.tolist() for k, v in cms.items()}}
            json.dump(step_out, open(of, "w"), indent=1)
            np.savez_compressed(of[:-5] + ".npz", **npz)
            f, r_, x = (step_out["splits"][k] for k in ["forget", "retained", "mixed"])
            print(f"[{a.ds} s{s} t{t}] {time.time() - t0:.0f}s | forget P {f['prompt_tok_acc']:.3f} "
                  f"Pc {f['prompt_content_tok_acc']:.3f} "
                  f"last {f['prompt_last_acc']:.3f} (eq_saved {f['prompt_last_eq_saved']:.3f}) G {f['gen_tok_acc']:.3f} | "
                  f"retain P {r_['prompt_tok_acc']:.3f} last {r_['prompt_last_acc']:.3f} G {r_['gen_tok_acc']:.3f} | "
                  f"mixed last {x['prompt_last_acc']:.3f}", flush=True)


if __name__ == "__main__":
    main()
