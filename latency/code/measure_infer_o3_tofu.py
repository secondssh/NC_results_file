#!/usr/bin/env python3
"""O3 추론 latency + 메모리 (TOFU) — LLM LoRA · OOD detector 모두 저장 환경 (o3 env, peft 0.10.0) 에서.

PTOFU_experiment 원래 평가 경로 (src/model/unlearn.py eval / get_ood_weights, src/metrics/PTofu.py) 와 같다:
  LLM : model_src.LlamaForCausalLM_ood (bf16, cuda:0) + init_olora / init_oodweight / init_active_adapters_d
        + model_src.PeftModel (stage_t/checkpoints = 마지막 stage adapter), lora_target_modules = adapter 의 7 종
        prompt = PTofu._build_prompt(tokenizer, question), greedy, bs1 (O3 는 OOD weight 가 문항 단위)
  OOD : model_src.RobertaForSelector_inference (roberta-large + stage_t LoRA), stage_t 통계 (cuda),
        ood_data.ptofu_tokenization → get_unsup_Mah_score_s → clr.decision_function (CPU) → run_ood.obtain_weights → max(0, w)
입력: measure_infer_v2.py (TOFU) 와 같은 200 문항, 같은 순서 (ptofu_SD step020 forget 100 + ptofu_RD step020 retain 100, seed 1234)
측정: OOD 200 문항 (문항마다 sync), LLM fixed64 (min = max = 64) 앞 24 문항 3 회, natural (원래 평가 max_new_tokens 200) 앞 48 문항 1 회.
  LLM 에는 문항별 OOD weight 를 그대로 넣음 (실제 O3 와 같은 LoRA 세기). TTFT = 첫 logits (LogitsProcessor).
출력: --out_dir/o3_tofu_t{t}.json
사용 (PTOFU_experiment 에서): CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 <o3 env python> .../measure_infer_o3_tofu.py --steps 0 19
"""
import argparse
import json
import os
import pickle
import random
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
REPO = os.path.dirname(MC)
PT = os.path.join(os.path.dirname(REPO), "PTOFU_experiment")
os.chdir(PT)
sys.path.insert(0, os.path.join(PT, "src"))
from transformers import AutoConfig, LogitsProcessor, LogitsProcessorList, RobertaTokenizer  # noqa: E402
from transformers.models.llama.configuration_llama import LlamaConfig  # noqa: E402


# PTOFU src/model/unlearn.py 와 같은 패치 (transformers 4.37 은 Llama-3.1 rope_scaling 형식을 거부 → 검사만 완화)
def _rope_scaling_validation_permissive(self):
    if self.rope_scaling is None:
        return
    if not isinstance(self.rope_scaling, dict):
        raise ValueError("`rope_scaling` must be a dictionary")
    if "type" not in self.rope_scaling and "rope_type" not in self.rope_scaling:
        raise ValueError("`rope_scaling` must contain a 'type' or 'rope_type' field")


LlamaConfig._rope_scaling_validation = _rope_scaling_validation_permissive
from model_src.modeling_llama_hacked_o import LlamaForCausalLM_ood  # noqa: E402
from model_src.peft_model_hacked_o import PeftModel  # noqa: E402
from model_src.ood_model_selector import RobertaForSelector_inference  # noqa: E402
from ood.run_ood import obtain_weights  # noqa: E402
from ood.ood_data import ptofu_tokenization  # noqa: E402
from metrics.PTofu import _build_prompt, _load_tokenizer  # noqa: E402

D = os.path.join(REPO, "data_new")
BASE = os.path.join(MC, "train31Llama/trained_model/O3_LLAMA31_TOFU")
LORA = os.path.join(PT, "files/results/unlearn_ptofu/llama31_8b_O3/seed_0/stage_{t}/checkpoints")
OOD = os.path.join(PT, "ood_checkpoints_seed_0/stage_{t}")


class FirstToken(LogitsProcessor):
    def __init__(self):
        self.t = None

    def __call__(self, input_ids, scores):
        if self.t is None:
            torch.cuda.synchronize(); self.t = time.perf_counter()
        return scores


def stats(xs):
    a = np.asarray(xs, float)
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "n": int(a.size)}


def pick_items(n):
    F = json.load(open(f"{D}/ptofu_SD/tofu_step020_SD.json"))
    R = json.load(open(f"{D}/ptofu_RD/tofu_step020_RD.json"))
    rng = random.Random(1234)
    items = [F[i] for i in sorted(rng.sample(range(len(F)), n))] + [R[i] for i in sorted(rng.sample(range(len(R)), n))]
    rng.shuffle(items)
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=[0, 19])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n_fixed", type=int, default=24)
    ap.add_argument("--n_natural", type=int, default=48)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results", "v2", "timing"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    import peft
    import transformers
    dev = torch.device("cuda")
    items = pick_items(a.n)
    qs = [x.get("question", x.get("instruction", "")) for x in items]
    ood_tok = RobertaTokenizer.from_pretrained("roberta-large")
    ood_inputs = ptofu_tokenization(items, ood_tok, is_paraphrase=False)
    assert len(ood_inputs) == len(items)
    tok = _load_tokenizer(BASE)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token; tok.pad_token_id = tok.eos_token_id
    prompts = [_build_prompt(tok, q) for q in qs]

    for t in a.steps:
        res = {"t": t, "lora": LORA.format(t=t), "ood": OOD.format(t=t), "gpu": torch.cuda.get_device_name(0),
               "env": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
               "n_items": len(items)}
        # --- OOD detector ---
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        p = OOD.format(t=t)
        om = RobertaForSelector_inference("roberta-large", lora_path=p + "_roberta", projection_dim=100).to(dev)
        clr = pickle.load(open(p + "_ocsvm.pkl", "rb"))
        gmm = pickle.load(open(p + "_gmm_w.pkl", "rb"))
        x0 = json.load(open(p + "_ocsvm_threshold.json"))[0]
        ls = [[m.to(dev) if torch.is_tensor(m) else m for m in torch.load(p + f"_{k}_list.pt", map_location="cpu")]
              for k in ("mean", "precision", "fea")]
        torch.cuda.synchronize()
        res["ood_weights_MB"] = torch.cuda.memory_allocated() / 2**20

        def score(x):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            mah = om.get_unsup_Mah_score_s(x, *ls, dev)[:, 1:]
            torch.cuda.synchronize(); t1 = time.perf_counter()
            w = max(obtain_weights(clr.decision_function(mah), gmm, None, x0), 0)
            t2 = time.perf_counter()
            return float(w), 1e3 * (t1 - t0), 1e3 * (t2 - t1)

        for x in ood_inputs[:2]:
            score(x)
        torch.cuda.reset_peak_memory_stats()
        recs = [score(x) for x in ood_inputs]
        ws = [r[0] for r in recs]
        res["ood"] = {"total_ms": stats([r[1] + r[2] for r in recs]), "roberta_mahalanobis_ms": stats([r[1] for r in recs]),
                      "ocsvm_weight_cpu_ms": stats([r[2] for r in recs]), "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
                      "mean_weight": float(np.mean(ws)), "note": "OOD 입력 토큰화(ptofu_tokenization) 는 원래 평가처럼 미리 해 둠"}
        print(f"[o3 tofu t{t}] OOD {res['ood']['total_ms']['mean']:.1f}±{res['ood']['total_ms']['std']:.1f} ms "
              f"(RoBERTa+Mah {res['ood']['roberta_mahalanobis_ms']['mean']:.1f}, OCSVM {res['ood']['ocsvm_weight_cpu_ms']['mean']:.1f}), "
              f"peak {res['ood']['peak_allocated_MB']:.0f} MB, mean w {res['ood']['mean_weight']:.3f}", flush=True)
        del om, ls; torch.cuda.empty_cache()

        # --- LLM ---
        torch.cuda.reset_peak_memory_stats()
        lora = LORA.format(t=t)
        cfg = AutoConfig.from_pretrained(BASE)
        cfg.lora_target_modules = json.load(open(os.path.join(lora, "adapter_config.json")))["target_modules"]
        cfg.orthogonal_loss, cfg.orthogonal_loss_weight = False, 0.1
        model = LlamaForCausalLM_ood.from_pretrained(BASE, config=cfg, torch_dtype=torch.bfloat16, device_map={"": 0})
        model.init_olora(orthogonal_loss=False, olora_weights=[])
        model.init_oodweight(ood_weight=[1, 1])
        model.init_active_adapters_d(active_adapters_d=["default"])
        model = PeftModel.from_pretrained(model, lora, is_trainable=True)
        model.eval()
        res["llm_weights_MB"] = torch.cuda.memory_allocated() / 2**20

        @torch.no_grad()
        def one(prompt, w, new, fixed):
            model.init_oodweight(ood_weight=[1, w])
            enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(dev)
            ft = FirstToken()
            kw = dict(min_new_tokens=new) if fixed else {}
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = model.generate(**enc, max_new_tokens=new, do_sample=False, eos_token_id=tok.eos_token_id,
                                 pad_token_id=tok.pad_token_id, logits_processor=LogitsProcessorList([ft]), **kw)
            torch.cuda.synchronize(); t1 = time.perf_counter()
            steps = out.size(1) - enc["input_ids"].size(1)
            return 1e3 * (t1 - t0), 1e3 * (ft.t - t0), 1e3 * (t1 - ft.t) / max(steps - 1, 1), steps, enc["input_ids"].size(1)

        def run(n, new, fixed, reps):
            for k in range(2):
                one(prompts[k], ws[k], new, fixed)
            torch.cuda.reset_peak_memory_stats()
            rr = [one(prompts[k], ws[k], new, fixed) for _ in range(reps) for k in range(n)]
            return {"ms_per_sample": stats([r[0] for r in rr]), "ttft_ms": stats([r[1] for r in rr]),
                    "decode_ms_per_token": stats([r[2] for r in rr]), "mean_new_tokens": float(np.mean([r[3] for r in rr])),
                    "mean_prompt_tokens": float(np.mean([r[4] for r in rr])), "n_samples": n, "reps": reps, "warmup": 2, "bs": 1,
                    "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20}

        res["fixed64_bs1"] = run(a.n_fixed, 64, True, 3)
        res["natural_bs1"] = run(a.n_natural, 200, False, 1)
        for k in ("fixed64_bs1", "natural_bs1"):
            r = res[k]
            print(f"[o3 tofu t{t}] {k}: {r['ms_per_sample']['mean']:.1f} ms (+OOD → {r['ms_per_sample']['mean'] + res['ood']['total_ms']['mean']:.1f}), "
                  f"TTFT {r['ttft_ms']['mean']:.2f} (+OOD → {r['ttft_ms']['mean'] + res['ood']['total_ms']['mean']:.2f}), "
                  f"decode {r['decode_ms_per_token']['mean']:.2f} ms/tok, {r['mean_new_tokens']:.1f} tok, peak {r['peak_allocated_MB']:.0f} MB", flush=True)
        json.dump(res, open(os.path.join(a.out_dir, f"o3_tofu_t{t}.json"), "w"), indent=1)
        del model; torch.cuda.empty_cache()
    print("[done]")


if __name__ == "__main__":
    main()
