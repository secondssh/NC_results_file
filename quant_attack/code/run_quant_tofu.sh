#!/usr/bin/env bash
# Quantization attack on TOFU continual unlearning — step t19 (AG20, seed 0), NF4 4bit.
#
#   표:            bf16(원래) | backbone 4bit | backbone + MoE 4bit
#     base          (소거 전 기준선 — 양자화 자체의 손실)
#     Ours (RULE)
#
# 인자/데이터는 modify_code/repro_check/run_tofu_repro_lunar.sh 의 t19 와 같다 (eval batch 8).
# 원래(bf16) 성능 = run_results/.../repro_tofu_lunar/tofu_moe_fixed_layer16_0 t19.
#
# 사용법
#   CUDA_VISIBLE_DEVICES=0 bash quant_attack/run_quant_tofu.sh
#   MODES="none backbone all"   IMPL=fake|bnb   METHODS="base ours"
#   SMOKE=16 ...                # forget/retain 을 앞 16개로 잘라 재현/속도만 확인
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"                       # ScienceQA_expriment
MODIFY_CODE="${REPO}/modify_code"
DATA="${REPO}/data_new"

PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
MODEL_FAMILY="${MODEL_FAMILY:-llama3.1-8b-instruct}"
BASE_MODEL="${BASE_MODEL:-${MODIFY_CODE}/train31Llama/trained_model/O3_LLAMA31_TOFU}"
LAYER_MODIFIED="${LAYER_MODIFIED:-[16]}"
SEED="${SEED:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"           # repro_tofu_lunar 와 같은 값 (config 기본)
EVAL_SPLITS="${EVAL_SPLITS:-[forget,retained,real_authors,world_facts]}"
MODES="${MODES:-backbone all}"
IMPL="${IMPL:-fake}"
read -r -a METHODS <<< "${METHODS:-base ours}"

STEP="${STEP:-19}"
DOMAIN="$(printf "AG%02d" $((STEP + 1)))"
STEP_NUM="$(printf "%03d" $((STEP + 1)))"
TASK_ID="tofu_t${STEP}_${DOMAIN}"

FORGET_TRAIN="${DATA}/ptofu_SD_only/tofu_${DOMAIN}_SD.json"
RETAIN_TRAIN="${DATA}/ptofu_RD/tofu_step${STEP_NUM}_RD.json"
SD_TEST="${DATA}/ptofu_SD/tofu_step${STEP_NUM}_SD.json"
RD_TEST="${DATA}/ptofu_RD/tofu_step${STEP_NUM}_RD.json"
UTILITY1="${DATA}/tofu_real_authors/real_authors_test.json"
UTILITY2="${DATA}/tofu_world_facts/world_facts_test.json"
RULE_CKPT="${RULE_CKPT:-${MODIFY_CODE}/outputs/tofu_ckpt_repro_lunar/seed_${SEED}/t${STEP}_${DOMAIN}/lora_moe.pt}"

OUT_ROOT="${OUT_ROOT:-${HERE}/results/tofu_t${STEP}_${IMPL}}"
HYDRA_ROOT="${HYDRA_ROOT:-${HERE}/hydra_outputs}"

if [[ -n "${SMOKE:-}" ]]; then
  SMOKE_DIR="${HERE}/.smoke_data"; mkdir -p "${SMOKE_DIR}"
  for var in SD_TEST RD_TEST; do
    src="${!var}"; dst="${SMOKE_DIR}/$(basename "${src}")"
    "${PY}" -c "import json,sys; d=json.load(open(sys.argv[1]))[:int(sys.argv[3])]; json.dump(d,open(sys.argv[2],'w'),ensure_ascii=False)" \
      "${src}" "${dst}" "${SMOKE}"
    printf -v "${var}" '%s' "${dst}"
  done
  OUT_ROOT="${OUT_ROOT}_smoke"
fi
mkdir -p "${OUT_ROOT}"

echo "[$(date)] quantization attack | TOFU t${STEP} (${DOMAIN})"
echo "  methods=${METHODS[*]}  modes=${MODES}  impl=${IMPL}  splits=${EVAL_SPLITS}  ckpt=${RULE_CKPT}"
echo "  -> ${OUT_ROOT}"

run_one () {   # $1=method $2=mode
  local method="$1" mode="$2" extra=()
  case "${method}" in
    base) ;;
    ours) [[ -f "${RULE_CKPT}" ]] || { echo "[SKIP] ours: ${RULE_CKPT} 없음"; return 0; }
          extra+=("+quant.moe_ckpt=${RULE_CKPT}") ;;
    *) echo "[ERR] unknown method ${method}"; return 1 ;;
  esac
  if [[ "${method}" != "ours" && "${mode}" == "all" ]]; then
    echo "[SKIP] ${method}: adapter 없음 (all == backbone)"; return 0
  fi
  local tag="${method}_${mode}" out="${OUT_ROOT}/${method}_${mode}.json"
  if [[ -f "${out}" && -n "${SKIP_EXISTING:-}" ]]; then echo "[SKIP] ${tag}: already done"; return 0; fi

  echo "================ ${tag} ================"
  ( cd "${MODIFY_CODE}" && "${PY}" "${HERE}/eval_quant_tofu.py" \
      --config-name forget \
      data_name=tofu \
      model_family="${MODEL_FAMILY}" \
      model_path="${BASE_MODEL}" \
      seed="${SEED}" \
      task_id="${TASK_ID}" \
      layer_modified="${LAYER_MODIFIED}" \
      data_path="${FORGET_TRAIN}" \
      r_data_path="${RETAIN_TRAIN}" \
      sampled_data_path="${SD_TEST}" \
      retain_data_path="${RD_TEST}" \
      utiliy_data_path1="${UTILITY1}" \
      utiliy_data_path2="${UTILITY2}" \
      load_unlearned_model=false \
      save_unlearned_model=false \
      save_path="${OUT_ROOT}" \
      eval_batch_size="${EVAL_BATCH_SIZE}" \
      hydra.run.dir="${HYDRA_ROOT}/${TASK_ID}/${tag}" \
      +quant.mode="${mode}" \
      +quant.impl="${IMPL}" \
      +quant.tag="${tag}" \
      +quant.eval_splits="${EVAL_SPLITS}" \
      +quant.out_file="${out}" \
      "${extra[@]}" )
}

for mode in ${MODES}; do
  for method in "${METHODS[@]}"; do
    run_one "${method}" "${mode}"
  done
done
echo "[$(date)] done -> ${OUT_ROOT}"
