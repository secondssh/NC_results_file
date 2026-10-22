#!/usr/bin/env bash
# F/R 구간 routing 측정 + 강제 routing 개입 (router_span.py). 사용 (modify_code 에서):
#   CUDA_VISIBLE_DEVICES=1 setsid nohup bash multi_attack/run_span.sh > multi_attack/logs/span.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
"${PY}" "${HERE}/router_span.py" --ds clinc --steps 4 8 --intervene none f2e0 2>&1 | grep --line-buffered -E "^\[|Error|Traceback"
"${PY}" "${HERE}/router_span.py" --ds tofu --steps 10 19 --intervene none f2F --max_items 200 2>&1 | grep --line-buffered -E "^\[|Error|Traceback"
"${PY}" "${HERE}/span_summary.py" && echo "[span] finished"
