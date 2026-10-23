#!/usr/bin/env bash
# RULE 학습 profile (GPU 1장 단독). 사용 (modify_code 에서):
#   CUDA_VISIBLE_DEVICES=1 setsid nohup bash latency/run_train_profile.sh > latency/logs/train_profile.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
for job in "clinc 0" "clinc 8" "tofu 0" "tofu 19"; do
  read -r ds t <<< "${job}"
  [[ -s "${HERE}/results/train_${ds}_t${t}.json" ]] && { echo "[skip] ${ds} t${t}"; continue; }
  echo "[$(date '+%F %T')] ${ds} t${t}"
  "${PY}" "${HERE}/profile_train.py" --ds "${ds}" --step "${t}" > "${HERE}/logs/train_${ds}_t${t}.log" 2>&1
  echo "  rc=$?"; grep -E "^\[phase\]|^\[done\]|Error|Traceback" "${HERE}/logs/train_${ds}_t${t}.log" | tail -15
done
echo "[$(date '+%F %T')] finished"
