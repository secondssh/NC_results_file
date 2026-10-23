#!/usr/bin/env bash
# 시간 측정 전용 재측정 (monitor --light: nvidia-smi + loadavg 만, 2초 간격 → 측정 대상 방해 없음). 메모리는 results/v2 (PSS) 값 사용.
# 모든 단계 다른 GPU idle (90초 연속 비어 있을 때 시작), GPU0.
# 결과: results/v2/timing/ (train_*.json, infer_*.json, o3_*.json, version_*.json, mon_*.json)
# 사용 (modify_code 에서): setsid nohup bash latency/run_v2_timing.sh > latency/logs/v2/timing.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$(dirname "${HERE}")")"
LP=/home/hai_ssh/.conda/envs/lunar/bin/python; OP=/home/hai_ssh/miniconda3/envs/o3/bin/python
R="${HERE}/results/v2/timing"; L="${HERE}/logs/v2/timing"; mkdir -p "${R}" "${L}"
G=/home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/unlearning_baseline_new/outputs/clincqa_fullft/ga_gd/seed_0/t0_work
log() { echo "[$(date '+%F %T')] $*"; }
wait_idle() { local ok=0; while (( ok < 4 )); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then ok=$((ok + 1)); else ok=0; fi
  (( ok < 4 )) && sleep 30; done; }
step() {  # $1 이름, $2 env 접두, 나머지 = 명령
  local name=$1 envs=$2; shift 2
  [[ -s "${R}/mon_${name}.json" ]] && { log "[skip] ${name}"; return; }
  wait_idle; log "START ${name}"
  ( cd "${REPO}" && env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 ${envs} "${LP}" "${HERE}/monitor.py" --light --interval 2 \
      --out "${R}/mon_${name}.json" -- "$@" ) > "${L}/${name}.log" 2>&1
  log "END ${name} rc=$? | $(grep -E '^\[monitor\]' "${L}/${name}.log" | tail -1)"
}
step train_clinc_t0_copy "" "${LP}" "${HERE}/profile_train.py" --ds clinc --step 0 --out_dir "${R}"
for job in "clinc 0" "clinc 8" "tofu 0" "tofu 19"; do
  read -r ds t <<< "${job}"
  step "train_${ds}_t${t}_nocopy" "" "${LP}" "${HERE}/profile_train.py" --ds "${ds}" --step "${t}" --no_deepcopy --out_dir "${R}"
done
step infer_clinc "" "${LP}" "${HERE}/measure_infer_v2.py" --ds clinc --steps 0 4 8 --gd_path "${G}" --out "${R}/infer_clinc.json"
step infer_tofu  "" "${LP}" "${HERE}/measure_infer_v2.py" --ds tofu --steps 0 10 19 --out "${R}/infer_tofu.json"
step o3_ood_clinc "" "${OP}" "${HERE}/measure_o3_ood.py" --steps 0 8 --out_dir "${R}"
step o3_llm_clinc "PYTHONPATH=/home/hai_ssh/pylibs/peft0181" "${LP}" "${HERE}/measure_infer_o3_llm.py" --steps 0 8 --out_dir "${R}"
step version_lunar "" "${LP}" "${HERE}/version_check.py" --tag lunar --out_dir "${R}"
step version_o3 "" "${OP}" "${HERE}/version_check.py" --tag o3 --out_dir "${R}"
log "timing finished | other GPU idle: $(grep -ho '"clean": [a-z]*' ${R}/mon_*.json | sort | uniq -c | tr '\n' ' ')"
