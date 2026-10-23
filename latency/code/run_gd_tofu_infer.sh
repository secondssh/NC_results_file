#!/usr/bin/env bash
# TOFU GD (ga_gd, t0_AG01, /data2 에서 전송) 추론 측정: run_o3_train.sh 끝난 뒤 ("o3 train finished"), GPU 2장 90초 연속 idle 일 때 GPU0.
#   1) 시간: monitor --light → results/v2/timing/infer_tofu.json 에 "gd" 추가
#   2) 메모리: monitor PSS → results/v2/infer_tofu.json 에 "gd" 추가
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$(dirname "${HERE}")")"
LP=/home/hai_ssh/.conda/envs/lunar/bin/python
G="${REPO}/unlearning_baseline_new/outputs/ptofu_fullft/ga_gd/seed_0/t0_AG01"
log() { echo "[$(date '+%F %T')] $*"; }
wait_idle() { local ok=0; while (( ok < 4 )); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then ok=$((ok + 1)); else ok=0; fi
  (( ok < 4 )) && sleep 30; done; }
until grep -q "o3 train finished" "${HERE}/logs/v2/o3_train.log" 2>/dev/null; do sleep 60; done
for mode in timing mem; do
  if [[ "${mode}" == timing ]]; then out="${HERE}/results/v2/timing/infer_tofu.json"; mopt="--light --interval 2"; mon="${HERE}/results/v2/timing/mon_infer_tofu_gd.json"
  else out="${HERE}/results/v2/infer_tofu.json"; mopt=""; mon="${HERE}/results/v2/mon_infer_tofu_gd.json"; fi
  wait_idle; log "START gd tofu ${mode}"
  ( cd "${REPO}/modify_code" && env CUDA_VISIBLE_DEVICES=0 "${LP}" "${HERE}/monitor.py" ${mopt} --out "${mon}" -- \
      "${LP}" "${HERE}/measure_infer_v2.py" --ds tofu --no_base --gd_path "${G}" --out "${out}" ) > "${HERE}/logs/v2/gd_tofu_${mode}.log" 2>&1
  log "END gd tofu ${mode} rc=$? | $(grep -E '^\[(tofu|monitor)' "${HERE}/logs/v2/gd_tofu_${mode}.log" | tail -5 | tr '\n' ' ')"
done
log "gd tofu finished"
