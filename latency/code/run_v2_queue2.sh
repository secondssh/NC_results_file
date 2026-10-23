#!/usr/bin/env bash
# latency v2 큐 (2): GPU 1장 측정은 GPU0 에서 바로 (GPU1 의 다른 실험과 동시 — monitor.py 가 그때 GPU1 사용량 기록),
# GPU 2장 필요한 baseline 만 두 GPU 가 다 빌 때까지 대기.
#   GPU0: RULE 학습 profile (CLINC t0 원래 코드 + deepcopy 제거판 CLINC t0/t8, TOFU t0/t19) → 버전 영향 (lunar, o3 env)
#   그 다음: baseline v2 (2 GPU)
# 사용 (modify_code 에서): setsid nohup bash latency/run_v2_queue2.sh > latency/logs/v2/queue2.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/hai_ssh/.conda/envs/lunar/bin/python; OP=/home/hai_ssh/miniconda3/envs/o3/bin/python
R="${HERE}/results/v2"; L="${HERE}/logs/v2"; mkdir -p "${R}" "${L}"
log() { echo "[$(date '+%F %T')] $*"; }
gpu0_busy() { local u; u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0); nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "${u}"; }
step() {  # $1 = 이름, 나머지 = 명령 (GPU0)
  local name=$1; shift
  [[ -s "${R}/mon_${name}.json" ]] && { log "[skip] ${name}"; return; }
  while gpu0_busy; do sleep 30; done
  log "START ${name}"
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 "${PY}" "${HERE}/monitor.py" --out "${R}/mon_${name}.json" -- "$@" > "${L}/${name}.log" 2>&1
  log "END ${name} rc=$? | $(grep -E '^\[monitor\]' "${L}/${name}.log" | tail -1)"
}
step train_clinc_t0_copy "${PY}" "${HERE}/profile_train.py" --ds clinc --step 0 --out_dir "${R}"
for job in "clinc 0" "clinc 8" "tofu 0" "tofu 19"; do
  read -r ds t <<< "${job}"
  step "train_${ds}_t${t}_nocopy" "${PY}" "${HERE}/profile_train.py" --ds "${ds}" --step "${t}" --no_deepcopy --out_dir "${R}"
done
step version_lunar "${PY}" "${HERE}/version_check.py" --tag lunar
step version_o3 "${OP}" "${HERE}/version_check.py" --tag o3
log "GPU0 단계 끝 → baseline v2 는 GPU 2장이 빌 때까지 대기"
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do sleep 60; done
log "START baseline v2 (2 GPU)"
bash "${HERE}/run_baseline_train_v2.sh" > "${L}/baseline.log" 2>&1
log "END baseline v2 | $(tail -1 "${L}/baseline.log")"
log "queue2 finished"
