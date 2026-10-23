#!/usr/bin/env python3
"""RULE 학습 단계별 시간 + GPU peak memory, expert / router 학습 분리. README "### latency 실험" 2).

원래 학습 스크립트(mmain_fixed.py / mmain_tofu_fixed.py)를 원래 인자 그대로 실행하고, 그 모듈 안의 단계 함수만
시간·메모리 기록 wrapper 로 감싼다 (계산 경로는 원래와 같음). 저장 안 함 (save_unlearned_model=false), 평가 전에 멈춤.
  단계: load_model / deepcopy(학습용 복사본) / load_dataset_to_get_direction / split_raw_dataset_for_forget /
        generate_candidate_directions / prepare_trainset (activation 추출) / MoE 준비(load+expand 또는 init)
  MoE 학습: 같은 train_loader 로 1 epoch 씩, 시작 상태(가중치·optimizer·scheduler·requires_grad)를 되돌려 가며
        joint (원래 학습) / expert (gate freeze, LoRA 만) / router (LoRA freeze, gate 만)  — train_moe.py 의 mode 인자
        원래 학습 = joint × num_epochs(20). expert / router 는 "Expert Training / Router Training" 분리용.
메모리: 단계마다 reset_peak_memory_stats 후 max_memory_allocated, 단계 뒤 memory_allocated. activation cache 크기.

출력: latency/results/train_{ds}_t{t}.json
사용 (modify_code 에서): CUDA_VISIBLE_DEVICES=1 python latency/profile_train.py --ds clinc --step 0
"""
import argparse
import copy as _copy
import hashlib
import importlib
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MC = os.path.dirname(HERE)
os.chdir(MC)
sys.path.insert(0, MC)

D = "/home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new"
CLINC_DOMAINS = ["work", "travel", "home", "banking", "credit_cards", "kitchen_and_dining",
                 "auto_and_commute", "utility", "small_talk"]


def overrides(ds, t):
    if ds == "clinc":
        dom, joined = CLINC_DOMAINS[t], "_".join(CLINC_DOMAINS[:t + 1])
        prev = f"./outputs/repro_clinc_lunar_bs64/seed_0/t{t - 1}_{CLINC_DOMAINS[t - 1]}/lora_moe.pt" if t else ""
        o = ["data_name=clincqa", "eval_batch_size=16", "model_path=./train31Llama/trained_model/O3_LLAMA31_ClincQA",
             f"task_id=clincqa_t{t}_{dom}", f"data_path={D}/clincqa_SD_only_200/clincqa_{dom}_train_SD.json",
             f"r_data_path={D}/clincqa_RD_only_200_c/clincqa_not_{joined}_train_RD.json",
             "use_unverified=false", "use_harmful=true",
             f"sampled_data_path={D}/clincqa_SD_10/clincqa_{joined}_test_SD.json",
             f"retain_data_path={D}/clincqa_RD_10/clincqa_not_{joined}_test_RD.json",
             f"utiliy_data_path1={D}/mrpcqa/mrpc_test.json", f"utiliy_data_path2={D}/rteqa/rte_validation.json"]
    else:
        dom, step = f"AG{t + 1:02d}", f"{t + 1:03d}"
        prev = f"./outputs/tofu_ckpt_repro_lunar/seed_0/t{t - 1}_AG{t:02d}/lora_moe.pt" if t else ""
        o = ["data_name=tofu", "model_path=./train31Llama/trained_model/O3_LLAMA31_TOFU", f"task_id=tofu_t{t}_{dom}",
             f"data_path={D}/ptofu_SD_only/tofu_{dom}_SD.json", f"r_data_path={D}/ptofu_RD/tofu_step{step}_RD.json",
             f"sampled_data_path={D}/ptofu_SD/tofu_step{step}_SD.json", f"retain_data_path={D}/ptofu_RD/tofu_step{step}_RD.json",
             f"utiliy_data_path1={D}/tofu_real_authors/real_authors_test.json",
             f"utiliy_data_path2={D}/tofu_world_facts/world_facts_test.json"]
    o += ["model_family=llama3.1-8b-instruct", "seed=0", "layer_modified=[16]",
          f"load_unlearned_model={'true' if t else 'false'}", f"load_unlearned_model_path={prev}",
          "save_unlearned_model=false", "save_unlearned_model_path=", f"save_folder=latency_profile/{ds}_t{t}"]
    return o


class Rec:
    def __init__(self):
        self.phases = []

    def wrap(self, name, fn):
        def w(*a, **k):
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            out = fn(*a, **k)
            torch.cuda.synchronize()
            self.phases.append({"phase": name, "sec": time.perf_counter() - t0,
                                "peak_allocated_MB": torch.cuda.max_memory_allocated() / 2**20,
                                "allocated_after_MB": torch.cuda.memory_allocated() / 2**20,
                                "peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20})
            print(f"[phase] {name}: {self.phases[-1]['sec']:.1f}s peak {self.phases[-1]['peak_allocated_MB']:.0f} MB", flush=True)
            return out
        return w


def tensor_bytes(x):
    if torch.is_tensor(x):
        return x.numel() * x.element_size(), str(x.device)
    if isinstance(x, (list, tuple)):
        tot, dev = 0, None
        for y in x:
            b, d = tensor_bytes(y); tot += b; dev = dev or d
        return tot, dev
    return 0, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, choices=["clinc", "tofu"])
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--epochs_each", type=int, default=1)
    ap.add_argument("--no_deepcopy", action="store_true",
                    help="학습용 8B 복사본 대신 같은 모델 사용 (MoE 는 activation cache 로 학습되고, 삽입은 추출이 끝난 뒤라 결과 동일)")
    ap.add_argument("--out_dir", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    out = os.path.join(a.out_dir, f"train_{a.ds}_t{a.step}{'_nocopy' if a.no_deepcopy else ''}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    mod = importlib.import_module("mmain_fixed" if a.ds == "clinc" else "mmain_tofu_fixed")
    rec = Rec()
    info = {"ds": a.ds, "t": a.step, "gpu": torch.cuda.get_device_name(0), "epochs_each": a.epochs_each,
            "no_deepcopy": a.no_deepcopy, "torch": torch.__version__}

    for name in ("load_model", "load_dataset_to_get_direction", "split_raw_dataset_for_forget",
                 "generate_candidate_directions", "load_lora_moe_only", "expand_lora_moe_experts",
                 "prepare_estimated_net_moe_lora_list"):
        if hasattr(mod, name):
            setattr(mod, name, rec.wrap(name, getattr(mod, name)))

    orig_prep = mod.prepare_trainset

    def prep(*args, **kw):
        res = rec.wrap("prepare_trainset (activation 추출)", orig_prep)(*args, **kw)
        b, dev = tensor_bytes(list(res))
        info["activation_cache"] = {"MB": b / 2**20, "device": dev,
                                    "n_forget": int(res[0][0].size(0)), "n_retain": int(res[2][0].size(0))}
        # deepcopy 유무로 학습 입력이 같은지 확인용 (시간 측정 밖)
        h = hashlib.sha256()
        for lst in res:
            for x in lst:
                h.update(x.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        info["activation_cache"]["sha256"] = h.hexdigest()
        return res
    mod.prepare_trainset = prep

    class _Copy:
        deepcopy = staticmethod(rec.wrap("deepcopy (학습용 모델 복사본)", (lambda m: m) if a.no_deepcopy else _copy.deepcopy))
    mod.copy = _Copy

    orig_train = mod.train_moe_multiple_layers

    def train(model_list, train_loader, optimizer, scheduler, device, current_forget_idx, num_epochs=20, **kw):
        info.update(full_num_epochs=num_epochs, n_train_batches=len(train_loader), train_batch_size=train_loader.batch_size,
                    n_train_samples=len(train_loader.dataset), n_experts=int(model_list[0].n_experts),
                    current_forget_idx=int(current_forget_idx))
        init_w = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in model_list]
        init_rg = [{n: p.requires_grad for n, p in m.named_parameters()} for m in model_list]
        init_opt, init_sch = _copy.deepcopy(optimizer.state_dict()), _copy.deepcopy(scheduler.state_dict())
        for mode in ("joint", "expert", "router"):
            for m, w, rg in zip(model_list, init_w, init_rg):
                m.load_state_dict(w)
                for n, p in m.named_parameters():
                    p.requires_grad = rg[n]
            optimizer.load_state_dict(init_opt); scheduler.load_state_dict(init_sch)
            rec.wrap(f"moe_train_{mode} ({a.epochs_each} epoch)", orig_train)(
                model_list, train_loader, optimizer, scheduler, device, current_forget_idx,
                num_epochs=a.epochs_each, mode=mode, **kw)
        return model_list
    mod.train_moe_multiple_layers = train

    def stop_eval(*args, **kw):
        print("[profile] 학습 단계 끝 — 평가 건너뜀", flush=True)
    mod.evaluate_and_save = stop_eval
    if hasattr(mod, "save_lora_moe_only"):
        mod.save_lora_moe_only = lambda *x, **k: None

    # hydra.main 은 cfg 를 직접 받으면 그대로 task 함수에 넘긴다 (원래 실행과 같은 config + 인자)
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=os.path.join(MC, "config"), version_base=None):
        cfg = compose(config_name="forget", overrides=overrides(a.ds, a.step))
    t0 = time.perf_counter()
    mod.train(cfg)
    info["total_wall_sec_to_end_of_training"] = time.perf_counter() - t0
    info["phases"] = rec.phases
    json.dump(info, open(out, "w"), indent=1)
    print(f"[done] {out}", flush=True)


if __name__ == "__main__":
    main()
