#!/usr/bin/env bash
# expert 수 E 확장 통합 표용: (1) 학습 1 stage (scale_train.py) → (2) 실제 생성 한 회차 (base + E 2…320, scale_experts.py --only_e2e)
# GPU0, monitor --light, 시작 전 GPU 2장 90초 연속 idle 확인. 사용 (modify_code 에서):
#   setsid nohup bash latency/run_scale_all.sh > latency/logs/v2/timing/scale_all.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY=/home/hai_ssh/.conda/envs/lunar/bin/python
R="${HERE}/results/v2/timing"; L="${HERE}/logs/v2/timing"
log() { echo "[$(date '+%F %T')] $*"; }
wait_idle() { local ok=0; while (( ok < 4 )); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then ok=$((ok + 1)); else ok=0; fi
  (( ok < 4 )) && sleep 30; done; }
cd "$(dirname "${HERE}")"
wait_idle; log "START scale_train"
CUDA_VISIBLE_DEVICES=0 "${PY}" "${HERE}/monitor.py" --light --interval 2 --out "${R}/mon_scale_train.json" -- "${PY}" "${HERE}/scale_train.py" > "${L}/scale_train.log" 2>&1
log "END scale_train rc=$? | $(grep -E '^\[E=|^\[monitor' "${L}/scale_train.log" | tr '\n' ' ')"
wait_idle; log "START scale_e2e"
CUDA_VISIBLE_DEVICES=0 "${PY}" "${HERE}/monitor.py" --light --interval 2 --out "${R}/mon_scale_e2e.json" -- \
  "${PY}" "${HERE}/scale_experts.py" --only_e2e --e2e_E 2 10 21 40 80 160 320 > "${L}/scale_e2e.log" 2>&1
log "END scale_e2e rc=$? | $(grep -E '^\[e2e|^\[monitor' "${L}/scale_e2e.log" | tr '\n' ' ')"
log "scale all finished"
