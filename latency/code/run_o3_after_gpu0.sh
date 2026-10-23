#!/usr/bin/env bash
# O3 CLINC 추론 측정 (GPU0): run_v2_queue2.sh 의 GPU0 단계가 끝난 뒤.
#   baseline 이 아직 시작 전이면 바로 (baseline 은 GPU 2장이 다 빌 때까지 기다리므로 O3 가 끝난 뒤 시작),
#   이미 시작했으면 queue2 가 끝난 뒤.
#   1) OOD 점수 계산: o3 env (detector 저장 환경, peft 0.10.0) → 문항별 OOD weight 저장
#   2) LLM 생성: lunar + peft 0.18.1 (LLM LoRA 저장 peft 버전), 1) 의 weight 사용
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$(dirname "${HERE}")")"
LP=/home/hai_ssh/.conda/envs/lunar/bin/python; OP=/home/hai_ssh/miniconda3/envs/o3/bin/python
R="${HERE}/results/v2"; L="${HERE}/logs/v2"; Q="${L}/queue2.log"
log() { echo "[$(date '+%F %T')] $*"; }
until grep -q "GPU0 단계 끝" "${Q}" 2>/dev/null; do sleep 30; done
if grep -q "START baseline" "${Q}"; then log "baseline 먼저 시작 → queue2 끝날 때까지 대기"; until grep -q "queue2 finished" "${Q}"; do sleep 60; done; fi
u=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0)
while nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "${u}"; do sleep 30; done
cd "${REPO}"
log "START o3 ood (o3 env)"
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 "${LP}" "${HERE}/monitor.py" --out "${R}/mon_o3_ood_clinc.json" -- \
  "${OP}" "${HERE}/measure_o3_ood.py" --steps 0 8 > "${L}/o3_ood_clinc.log" 2>&1
log "END o3 ood rc=$? | $(grep -E '^\[(o3|monitor)' "${L}/o3_ood_clinc.log" | tail -3 | tr '\n' ' ')"
log "START o3 llm (lunar + peft 0.18.1)"
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/hai_ssh/pylibs/peft0181 "${LP}" "${HERE}/monitor.py" --out "${R}/mon_o3_llm_clinc.json" -- \
  "${LP}" "${HERE}/measure_infer_o3_llm.py" --steps 0 8 > "${L}/o3_llm_clinc.log" 2>&1
log "END o3 llm rc=$? | $(grep -E '^\[(o3|monitor)' "${L}/o3_llm_clinc.log" | tail -5 | tr '\n' ' ')"
log "o3 finished"
