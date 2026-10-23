#!/usr/bin/env bash
# baseline (2 GPU) → O3 추론 (GPU0). 사용 (modify_code 에서):
#   setsid nohup bash latency/run_baseline_then_o3.sh > latency/logs/after_rule.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$(dirname "${HERE}")")"
echo "[$(date '+%F %T')] baseline"
bash "${HERE}/run_baseline_train.sh" > "${HERE}/logs/baseline.log" 2>&1
echo "[$(date '+%F %T')] baseline 끝 → O3 추론 (GPU0)"
( cd "${REPO}" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/hai_ssh/pylibs/peft0181 /home/hai_ssh/.conda/envs/lunar/bin/python \
    "${HERE}/measure_infer_o3.py" --steps 0 8 ) > "${HERE}/logs/infer_o3.log" 2>&1
echo "[$(date '+%F %T')] O3 rc=$?"; grep -E "^\[o3|Traceback|Error" "${HERE}/logs/infer_o3.log" | tail -12
echo "[$(date '+%F %T')] all finished"
