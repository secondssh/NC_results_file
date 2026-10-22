#!/usr/bin/env bash
# RULE TOFU t0, t10 — CLINC 와 같은 설정 (fake NF4 4bit, backbone + all, eval bs 8 = repro_tofu_lunar), GPU1
cd /home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/quant_attack
export CUDA_VISIBLE_DEVICES=1 SKIP_EXISTING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for s in 0 10; do
  echo "[$(date '+%F %T')] TOFU t${s} 시작"
  STEP=${s} MODES="backbone all" METHODS=ours bash run_quant_tofu.sh > logs/tofu_t${s}_rule.log 2>&1
  echo "[$(date '+%F %T')] TOFU t${s} 끝 rc=$?"
done
