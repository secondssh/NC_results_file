#!/usr/bin/env bash
# CLINC t0 GA+GD(full-FT) quantization: run_quant_t0_clinc.sh(RULE→O3) 끝나면 GPU1 에서 none + backbone(=전체 4bit).
# full-FT 라 adapter 가 없어 all == backbone. eval batch 8 = baseline 원래 평가와 같음. none 은 같은 스크립트로 잰 bf16 기준.
cd /home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/quant_attack
while pgrep -f run_quant_t0_clinc.sh > /dev/null; do sleep 60; done
echo "[$(date '+%F %T')] GD t0 시작"
CUDA_VISIBLE_DEVICES=1 SKIP_EXISTING=1 STEP=0 DOMAIN=work JOINED=work EVAL_BATCH_SIZE=8 \
  OUT_ROOT=$PWD/results/clinc_t0_gd_bs8_fake METHODS=ga_gd MODES="none backbone" \
  bash run_quant_clinc.sh > logs/t0_gd.log 2>&1
echo "[$(date '+%F %T')] GD t0 끝 rc=$?"
