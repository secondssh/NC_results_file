#!/usr/bin/env bash
# SMOKE=4 로 전 시점/세트 파이프라인 점검 → 실패 0 이면 본 실험(전체 문항) 시작.
# 사용 (modify_code 에서): CUDA_VISIBLE_DEVICES=1 setsid nohup bash multi_attack/run_smoke_then_full.sh > multi_attack/logs/run.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${HERE}/logs"
SMOKE=4 bash "${HERE}/run_cross_eval.sh" > "${HERE}/logs/smoke_run.log" 2>&1
if ! grep -q "finished (fail=0)" "${HERE}/logs/smoke_run.log"; then
  echo "[$(date '+%F %T')] SMOKE 실패 → 본 실험 시작 안 함 (${HERE}/logs/smoke_run.log)"; exit 1
fi
echo "[$(date '+%F %T')] SMOKE OK → full run"
bash "${HERE}/run_cross_eval.sh"
