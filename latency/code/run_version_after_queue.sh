#!/usr/bin/env bash
# run_v2_queue.sh 가 끝나면 (로그의 "queue finished" 로 확인) GPU0 에서 버전 영향 측정: lunar → o3 env.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; L="${HERE}/logs/v2"; R="${HERE}/results/v2"
LP=/home/hai_ssh/.conda/envs/lunar/bin/python; OP=/home/hai_ssh/miniconda3/envs/o3/bin/python
until grep -q "queue finished" "${L}/queue.log" 2>/dev/null; do sleep 60; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 30; done
for tag in lunar o3; do
  PY="${LP}"; [[ "${tag}" == o3 ]] && PY="${OP}"
  echo "[$(date '+%F %T')] version ${tag}"
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 "${LP}" "${HERE}/monitor.py" --out "${R}/mon_version_${tag}.json" -- \
    "${PY}" "${HERE}/version_check.py" --tag "${tag}" > "${L}/version_${tag}.log" 2>&1
  echo "  rc=$?"; grep -E "^\[(lunar|o3|monitor)|Traceback|Error" "${L}/version_${tag}.log" | tail -6
done
echo "[$(date '+%F %T')] version finished"
