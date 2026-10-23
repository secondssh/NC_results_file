#!/usr/bin/env bash
# latency v2 순차 큐 (한 번에 한 측정, 측정 GPU 외 다른 GPU idle). 단계마다 GPU 에 다른 프로세스가 없을 때 시작.
#   1) 추론 v2 CLINC (base, RULE t0/t4/t8, GD t0)   GPU0
#   2) 추론 v2 TOFU  (base, RULE t0/t10/t19)          GPU0
#   3) RULE 학습 profile: CLINC t0 (원래 코드, deepcopy) + deepcopy 제거판 CLINC t0/t8, TOFU t0/t19   GPU0
#   4) baseline GA / GD / NPO+KL 학습 v2 (2 GPU)
# 모든 단계 monitor.py 로 감쌈 (nvidia-smi peak / host RSS peak / load average).
# 사용 (modify_code 에서): setsid nohup bash latency/run_v2_queue.sh > latency/logs/v2/queue.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/hai_ssh/.conda/envs/lunar/bin/python
R="${HERE}/results/v2"; L="${HERE}/logs/v2"; mkdir -p "${R}" "${L}"
G=/home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/unlearning_baseline_new/outputs/clincqa_fullft/ga_gd/seed_0/t0_work
log() { echo "[$(date '+%F %T')] $*"; }
wait_idle() { while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do log "GPU 사용 중 — 대기"; sleep 60; done; }
step() {  # $1 = 이름, 나머지 = 명령 (GPU0)
  local name=$1; shift
  [[ -s "${R}/mon_${name}.json" ]] && { log "[skip] ${name}"; return; }
  wait_idle; log "START ${name}"
  CUDA_VISIBLE_DEVICES=0 "${PY}" "${HERE}/monitor.py" --out "${R}/mon_${name}.json" -- "$@" > "${L}/${name}.log" 2>&1
  log "END ${name} rc=$? | $(grep -E '^\[monitor\]' "${L}/${name}.log" | tail -1)"
}
step infer_clinc "${PY}" "${HERE}/measure_infer_v2.py" --ds clinc --steps 0 4 8 --gd_path "${G}"
step infer_tofu  "${PY}" "${HERE}/measure_infer_v2.py" --ds tofu --steps 0 10 19
step train_clinc_t0_copy "${PY}" "${HERE}/profile_train.py" --ds clinc --step 0 --out_dir "${R}"
for job in "clinc 0" "clinc 8" "tofu 0" "tofu 19"; do
  read -r ds t <<< "${job}"
  step "train_${ds}_t${t}_nocopy" "${PY}" "${HERE}/profile_train.py" --ds "${ds}" --step "${t}" --no_deepcopy --out_dir "${R}"
done
wait_idle; log "START baseline v2 (2 GPU)"
bash "${HERE}/run_baseline_train_v2.sh" > "${L}/baseline.log" 2>&1
log "END baseline v2 | $(tail -1 "${L}/baseline.log")"
log "queue finished"
