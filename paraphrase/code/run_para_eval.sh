#!/usr/bin/env bash
# Paraphrase 평가 작업자: seed0 step별 가중치로 SD(forget)/RD(retained)만 평가 (eval_only, 학습 없음)
# 평가 설정은 원래 평가와 동일 (전체 문항, config 기본 eval batch: TOFU 8 / CLINC 16, 생성 설정 동일). util은 평가 안 함.
# 작업 = (유형 × dataset × step). 결과가 이미 있거나 다른 작업자가 잡은(락) 작업은 건너뜀 → GPU마다 작업자를 여러 개 띄워도 됨
#   DATASETS: 이 작업자가 맡을 데이터셋 ("tofu" / "clinc" / "tofu clinc"), CLINC는 mmain_fixed.py eval_only 추가 후 사용
# (modify_code에서) CUDA_VISIBLE_DEVICES=0 DATASETS="tofu" setsid nohup bash repro_check/run_para_eval.sh > run_logs/para_eval/worker_gpu0_a.log 2>&1 < /dev/null & disown
set -uo pipefail

PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
DATASETS="${DATASETS:?DATASETS 필요 (tofu / clinc / \"tofu clinc\")}"
TYPES="${TYPES:-para para_ctx para_struct para_bt para_noise}"
WORKER="gpu${CUDA_VISIBLE_DEVICES:-x}_$$"
ROOT="/home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/data_new"
MODEL_FAMILY="llama3.1-8b-instruct"
RES_ROOT="./run_results/eval_basemoe_error/${MODEL_FAMILY}/para_eval"
LOCK_ROOT="${RES_ROOT}/_locks"
LOG_ROOT="./run_logs/para_eval"
mkdir -p "${LOCK_ROOT}" "${LOG_ROOT}"
export O3_RUN_MODE=eval_only O3_EVAL_SPLITS=forget,retained
unset MOE_WRAP MOE_TRAIN_BS

TOFU_DOMAINS=($(for i in $(seq 1 20); do printf "AG%02d " "$i"; done))
CLINC_DOMAINS=("work" "travel" "home" "banking" "credit_cards" "kitchen_and_dining" "auto_and_commute" "utility" "small_talk")

log() { echo "[$(date '+%F %T')][${WORKER}] $*"; }

# 작업 목록 (우선순위 순): 유형별 CLINC t0–t8 + TOFU t0–t9, 그다음 유형별 TOFU t10–t19
JOBS=()
for ty in ${TYPES}; do
  for t in $(seq 0 8); do JOBS+=("clinc ${ty} ${t}"); done
  for t in $(seq 0 9); do JOBS+=("tofu ${ty} ${t}"); done
done
for ty in ${TYPES}; do for t in $(seq 10 19); do JOBS+=("tofu ${ty} ${t}"); done; done

job_paths() {  # $1=ds $2=type $3=t → task_id, save_folder, result 경로 설정
  local ds=$1 ty=$2 t=$3
  if [[ "${ds}" == "tofu" ]]; then
    domain="${TOFU_DOMAINS[$t]}"; task_id="tofu_t${t}_${domain}"
    save_folder="para_eval/tofu_${ty}/tofu_moe_fixed_layer16_0/lunar_${MODEL_FAMILY}_eval_base_${task_id}"
  else
    domain="${CLINC_DOMAINS[$t]}"; task_id="clincqa_t${t}_${domain}"
    save_folder="para_eval/clinc_${ty}/clincqa_moe_fixed_layer16_0/lunar_${MODEL_FAMILY}_eval_base_${task_id}"
  fi
  result="./run_results/eval_basemoe_error/${MODEL_FAMILY}/${save_folder}/forget_${task_id}.json"
}

run_tofu() {  # run_tofu_repro_lunar.sh 와 같은 인자, 평가 데이터만 _{type} 폴더, seed0 ckpt
  local ty=$1 t=$2 step_num; step_num=$(printf "%03d" $((t + 1)))
  ${PY} mmain_tofu_fixed.py --config-name forget \
    data_name="tofu" model_family="${MODEL_FAMILY}" model_path="./train31Llama/trained_model/O3_LLAMA31_TOFU" \
    seed=0 task_id="${task_id}" layer_modified="[16]" \
    data_path="${ROOT}/ptofu_SD_only/tofu_${domain}_SD.json" \
    r_data_path="${ROOT}/ptofu_RD/tofu_step${step_num}_RD.json" \
    sampled_data_path="${ROOT}/ptofu_SD_${ty}/tofu_step${step_num}_SD.json" \
    retain_data_path="${ROOT}/ptofu_RD_${ty}/tofu_step${step_num}_RD.json" \
    utiliy_data_path1="${ROOT}/tofu_real_authors/real_authors_test.json" \
    utiliy_data_path2="${ROOT}/tofu_world_facts/world_facts_test.json" \
    load_unlearned_model=false \
    load_unlearned_model_path="./outputs/tofu_ckpt_repro_lunar/seed_0/t${t}_${domain}/lora_moe.pt" \
    save_unlearned_model=false \
    save_folder="${save_folder}"
}

run_clinc() {  # run_clinc_repro_other.sh 와 같은 인자 (eval_batch_size=16), 평가 데이터만 _{type} 폴더, seed0 ckpt
  local ty=$1 t=$2 joined
  joined="$(IFS=_; echo "${CLINC_DOMAINS[*]:0:$((t + 1))}")"
  ${PY} mmain_fixed.py --config-name forget \
    data_name="clincqa" eval_batch_size=16 model_family="${MODEL_FAMILY}" \
    model_path="./train31Llama/trained_model/O3_LLAMA31_ClincQA" \
    seed=0 task_id="${task_id}" layer_modified="[16]" \
    data_path="${ROOT}/clincqa_SD_only_200/clincqa_${domain}_train_SD.json" \
    r_data_path="${ROOT}/clincqa_RD_only_200_c/clincqa_not_${joined}_train_RD.json" \
    use_unverified=false use_harmful=true \
    sampled_data_path="${ROOT}/clincqa_SD_10_${ty}/clincqa_${joined}_test_SD.json" \
    retain_data_path="${ROOT}/clincqa_RD_10_${ty}/clincqa_not_${joined}_test_RD.json" \
    utiliy_data_path1="${ROOT}/mrpcqa/mrpc_test.json" \
    utiliy_data_path2="${ROOT}/rteqa/rte_validation.json" \
    load_unlearned_model=false \
    load_unlearned_model_path="./outputs/repro_clinc_lunar_bs64/seed_0/t${t}_${domain}/lora_moe.pt" \
    save_unlearned_model=false \
    save_folder="${save_folder}"
}

log "start DATASETS=[${DATASETS}] TYPES=[${TYPES}] PY=${PY}"
done_n=0; fail_n=0
for job in "${JOBS[@]}"; do
  read -r ds ty t <<< "${job}"
  [[ " ${DATASETS} " == *" ${ds} "* ]] || continue
  job_paths "${ds}" "${ty}" "${t}"
  [[ -s "${result}" ]] && continue
  lock="${LOCK_ROOT}/${ds}_${ty}_t${t}"
  mkdir "${lock}" 2>/dev/null || continue
  jlog="${LOG_ROOT}/${ds}_${ty}_t${t}.log"
  log "EVAL ${ds} ${ty} t${t} -> ${jlog}"
  start=$(date +%s)
  if [[ "${ds}" == "tofu" ]]; then run_tofu "${ty}" "${t}" > "${jlog}" 2>&1; else run_clinc "${ty}" "${t}" > "${jlog}" 2>&1; fi
  rc=$?
  if [[ ${rc} -eq 0 && -s "${result}" ]]; then
    done_n=$((done_n + 1)); log "DONE ${ds} ${ty} t${t} ($(( ($(date +%s) - start) / 60 )) min)"
  else
    fail_n=$((fail_n + 1)); log "FAIL ${ds} ${ty} t${t} rc=${rc} (락 해제, ${jlog} 확인)"; rmdir "${lock}"
  fi
done
log "finished: done ${done_n}, failed ${fail_n}"
