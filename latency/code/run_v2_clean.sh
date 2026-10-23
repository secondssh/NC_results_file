#!/usr/bin/env bash
# latency v2 깨끗한 재측정 (모든 단계 다른 GPU idle). 단계 시작 조건: GPU 2장이 90초 연속 비어 있음
# (다른 실험이 split 사이에 잠깐 GPU 를 비우는 순간 시작하는 문제 방지). 겹침 여부는 monitor.py 의 clean 필드로 확인.
#   1) baseline GA / GD / NPO+KL (2 GPU)
#   2) RULE 학습 profile: CLINC t0 원래 코드 + deepcopy 제거판 CLINC t0/t8, TOFU t0/t19 (GPU0)
#   3) 추론 v2 CLINC (base, RULE t0/t4/t8, GD) / TOFU (base, RULE t0/t10/t19) (GPU0)
#   4) O3 CLINC: OOD (o3 env) → LLM (lunar + peft 0.18.1) (GPU0)
#   5) 버전 영향 lunar / o3 env (GPU0)
# 사용 (modify_code 에서): setsid nohup bash latency/run_v2_clean.sh > latency/logs/v2/clean.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$(dirname "${HERE}")")"
LP=/home/hai_ssh/.conda/envs/lunar/bin/python; OP=/home/hai_ssh/miniconda3/envs/o3/bin/python
R="${HERE}/results/v2"; L="${HERE}/logs/v2"; mkdir -p "${R}" "${L}"
G=/home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/unlearning_baseline_new/outputs/clincqa_fullft/ga_gd/seed_0/t0_work
log() { echo "[$(date '+%F %T')] $*"; }
wait_idle() {  # GPU 2장이 90초 연속 비어 있을 때까지
  local ok=0
  while (( ok < 4 )); do
    if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then ok=$((ok + 1)); else ok=0; fi
    (( ok < 4 )) && sleep 30
  done
}
step() {  # $1 이름, $2 env 접두 (예: "PYTHONPATH=..."), 나머지 = 명령 (GPU0)
  local name=$1 envs=$2; shift 2
  [[ -s "${R}/mon_${name}.json" ]] && { log "[skip] ${name}"; return; }
  wait_idle; log "START ${name}"
  ( cd "${REPO}" && env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 ${envs} "${LP}" "${HERE}/monitor.py" --out "${R}/mon_${name}.json" -- "$@" ) > "${L}/${name}.log" 2>&1
  log "END ${name} rc=$? | $(grep -E '^\[monitor\]' "${L}/${name}.log" | tail -1)"
}
wait_idle; log "START baseline (2 GPU)"
bash "${HERE}/run_baseline_train_v2.sh" > "${L}/baseline_clean.log" 2>&1
log "END baseline | $(grep -c 'clean=True' "${L}/baseline_clean.log") / 6 clean"
step train_clinc_t0_copy "" "${LP}" "${HERE}/profile_train.py" --ds clinc --step 0 --out_dir "${R}"
for job in "clinc 0" "clinc 8" "tofu 0" "tofu 19"; do
  read -r ds t <<< "${job}"
  step "train_${ds}_t${t}_nocopy" "" "${LP}" "${HERE}/profile_train.py" --ds "${ds}" --step "${t}" --no_deepcopy --out_dir "${R}"
done
step infer_clinc "" "${LP}" "${HERE}/measure_infer_v2.py" --ds clinc --steps 0 4 8 --gd_path "${G}"
step infer_tofu  "" "${LP}" "${HERE}/measure_infer_v2.py" --ds tofu --steps 0 10 19
step o3_ood_clinc "" "${OP}" "${HERE}/measure_o3_ood.py" --steps 0 8
step o3_llm_clinc "PYTHONPATH=/home/hai_ssh/pylibs/peft0181" "${LP}" "${HERE}/measure_infer_o3_llm.py" --steps 0 8
step version_lunar "" "${LP}" "${HERE}/version_check.py" --tag lunar
step version_o3 "" "${OP}" "${HERE}/version_check.py" --tag o3
log "clean finished | clean flags: $(grep -ho '"clean": [a-z]*' ${R}/mon_*.json | sort | uniq -c | tr '\n' ' ')"
