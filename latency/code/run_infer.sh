#!/usr/bin/env bash
# 추론 latency 측정 (GPU 1장 단독). 사용 (modify_code 에서):
#   CUDA_VISIBLE_DEVICES=0 setsid nohup bash latency/run_infer.sh > latency/logs/infer.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
echo "[$(date '+%F %T')] start GPU=${CUDA_VISIBLE_DEVICES:-?}"
"${PY}" "${HERE}/measure_infer.py" --ds clinc --steps 0 4 8 2>&1 | grep --line-buffered -E "^\[|Error|Traceback"
"${PY}" "${HERE}/measure_infer.py" --ds tofu --steps 0 10 19 2>&1 | grep --line-buffered -E "^\[|Error|Traceback"
echo "[$(date '+%F %T')] finished"
