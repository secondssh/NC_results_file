#!/usr/bin/env bash
# CLINC t0 (work) quantization attack: RULE vs O3, NF4 fake quant, GPU1.
#   RULE: 재현 확인(none, 32개) → backbone, all      결과 results/clinc_t0_bs64_fake/
#   O3  : (detector = recent, 2026-10-21 확인) backbone, all   결과 results/o3_clinc_t0_oodrecent/
# 실행: setsid nohup bash quant_attack/run_quant_t0_clinc.sh > quant_attack/logs/t0_clinc.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
RES="${REPO}/modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct"
LOG="${HERE}/logs"
export CUDA_VISIBLE_DEVICES=1 SKIP_EXISTING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STEP=0 DOMAIN=work JOINED=work
log() { echo "[$(date '+%F %T')] $*"; }

log "[RULE t0] 재현 확인"
SMOKE=32 MODES=none METHODS=ours bash "${HERE}/run_quant_clinc.sh" > "${LOG}/t0_rule_smoke.log" 2>&1
if "${PY}" "${HERE}/check_rule_repro.py" "${HERE}/results/clinc_t0_bs64_fake_smoke/ours_none.json" \
     "${RES}/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0/lunar_llama3.1-8b-instruct_eval_base_clincqa_t0_work/forget_clincqa_t0_work.json" \
     | tee -a "${LOG}/t0_rule_smoke.log" | grep -q REPRO_OK; then
  log "[RULE t0] 재현 OK → backbone, all"
  MODES="backbone all" METHODS=ours bash "${HERE}/run_quant_clinc.sh" > "${LOG}/t0_rule.log" 2>&1
else
  log "[RULE t0] 재현 실패 → 건너뜀 (${LOG}/t0_rule_smoke.log)"
fi
log "[O3 t0] backbone, all"
OOD_DET=recent MODES="backbone all" bash "${HERE}/run_quant_o3_clinc.sh" > "${LOG}/t0_o3.log" 2>&1
log "끝"
