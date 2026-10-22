#!/bin/bash
# run_router_token_eval.sh(pid $1) 종료 후 빠진 step 채우기 + 요약
cd /home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code
while kill -0 "$1" 2>/dev/null; do sleep 60; done
DATASETS="clinc tofu" bash repro_check/run_router_token_eval.sh
/home/hai_ssh/.conda/envs/lunar/bin/python repro_check/summarize_router_token.py
echo "[$(date '+%F %T')] all done"
