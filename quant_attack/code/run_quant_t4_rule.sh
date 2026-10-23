#!/usr/bin/env bash
# CLINC t4 (credit_cards) RULE quantization attack (GPU1).
#   t0 / t8 과 같은 설정: NF4 fake quant, eval bs 16, ckpt outputs/repro_clinc_lunar_bs64/seed_0/t4_credit_cards
#   재현 확인(none, 32개) → backbone, all      결과 results/clinc_t4_bs64_fake/
# 실행: setsid nohup bash quant_attack/run_quant_t4_rule.sh > quant_attack/logs/t4_rule.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
RES="${REPO}/modify_code/run_results/eval_basemoe_error/llama3.1-8b-instruct"
LOG="${HERE}/logs"
log() { echo "[$(date '+%F %T')] $*"; }


export CUDA_VISIBLE_DEVICES=1 SKIP_EXISTING=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export STEP=4 DOMAIN=credit_cards JOINED=work_travel_home_banking_credit_cards
log "[RULE t4] 재현 확인"
SMOKE=32 MODES=none METHODS=ours bash "${HERE}/run_quant_clinc.sh" > "${LOG}/t4_rule_smoke.log" 2>&1
if "${PY}" "${HERE}/check_rule_repro.py" "${HERE}/results/clinc_t4_bs64_fake_smoke/ours_none.json" \
     "${RES}/repro_clinc_lunar_bs64/clincqa_moe_fixed_layer16_0/lunar_llama3.1-8b-instruct_eval_base_clincqa_t4_credit_cards/forget_clincqa_t4_credit_cards.json" \
     | tee -a "${LOG}/t4_rule_smoke.log" | grep -q REPRO_OK; then
  log "[RULE t4] 재현 OK → backbone, all"
  MODES="backbone all" METHODS=ours bash "${HERE}/run_quant_clinc.sh" > "${LOG}/t4_rule_run.log" 2>&1
  log "[RULE t4] rc=$?"
else
  log "[RULE t4] 재현 실패 → 중단 (${LOG}/t4_rule_smoke.log)"
  exit 1
fi
log "끝"
cd "${REPO}/modify_code/readme_main_results/_build" && "${PY}" build_all.py quant && log "quant 표 재생성 끝"
