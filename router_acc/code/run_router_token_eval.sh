#!/bin/bash
# router_token_eval.py 를 clinc → tofu 순서로 seed 0,1,2 실행 (이미 있는 step json 은 skip)
# CUDA_VISIBLE_DEVICES=1 setsid nohup bash repro_check/run_router_token_eval.sh > run_logs/router_token/run.log 2>&1 < /dev/null & disown
cd /home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
for ds in ${DATASETS:-clinc tofu}; do
  echo "[$(date '+%F %T')] start ${ds}"
  ${PY} repro_check/router_token_eval.py --ds "${ds}" --seeds 0 1 2 2>&1 | grep -v -E "FutureWarning|torch.load\(|payload = torch.load|return_legacy_cache"
  echo "[$(date '+%F %T')] end ${ds} rc=${PIPESTATUS[0]}"
done
