#!/usr/bin/env bash
# Quantization attack on CLINC continual unlearning — step t8 (seed 0), NF4 4bit.
#
#   표:            bf16(원래) | backbone 4bit | backbone + MoE 4bit
#     base          (소거 전 기준선 — 양자화 자체의 손실을 재는 행)
#     Ours (RULE)
#   (GA/NPO 는 METHODS 로 추가 가능. O3 는 run_quant_o3_clinc.sh)
#
# base 행이 핵심이다. 이게 없으면 "forget 이 되살아났다" 와 "그냥 모델이 망가졌다" 를
# 구분할 수 없다.
#
# backbone 모드: backbone 만 4bit, lora_A/lora_B/gate 는 bf16. 단 LUNAR_LoRA_MoE_net 안의
# backbone 사본(self.linear) 은 backbone 이므로 같이 NF4 로 만든다 (quant_utils.py 설명 참고).
# all 모드: 여기에 MoE 의 lora_A/lora_B/gate 까지 NF4 (fake_quant.py).
#
# 사용법
#   CUDA_VISIBLE_DEVICES=1 bash quant_attack/run_quant_clinc.sh
#   MODES="none backbone all"   # none=비양자화(재현 확인), backbone=백본만 4bit, all=백본+MoE(lora_A/B, gate) 4bit
#   IMPL=fake|bnb               # fake(기본): NF4 양자화 후 bf16 복원 / bnb: bitsandbytes 4bit 로드(예전 방식)
#   METHODS="base ours" ...     # base 는 MoE 가 없어 all == backbone 이므로 all 은 건너뜀
#   SMOKE=32 ...                # forget/retain 을 앞 32개로 잘라 속도/재현만 확인
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"                       # ScienceQA_expriment
MODIFY_CODE="${REPO}/modify_code"
BASELINE_DIR="${REPO}/unlearning_baseline_new"

PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"

MODEL_FAMILY="${MODEL_FAMILY:-llama3.1-8b-instruct}"
BASE_MODEL="${BASE_MODEL:-${MODIFY_CODE}/train31Llama/trained_model/O3_LLAMA31_ClincQA}"
LAYER_MODIFIED="${LAYER_MODIFIED:-[16]}"
SEED="${SEED:-0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"          # repro_clinc_lunar_bs64 (원래 성능) 와 같은 값. 바꾸면 비교 깨짐.
EVAL_SPLITS="${EVAL_SPLITS:-[forget,retained,mrpc,rte]}"
MODES="${MODES:-backbone all}"
IMPL="${IMPL:-fake}"
METHODS_STR="${METHODS:-base ours}"
read -r -a METHODS <<< "${METHODS_STR}"

# ---- t8 (= 9개 도메인 전부 소거된 최종 시점) ----
STEP="${STEP:-8}"
DOMAIN="${DOMAIN:-small_talk}"
JOINED="${JOINED:-work_travel_home_banking_credit_cards_kitchen_and_dining_auto_and_commute_utility_small_talk}"
TASK_ID="clincqa_t${STEP}_${DOMAIN}"

FORGET_TRAIN="${REPO}/data_new/clincqa_SD_only_200/clincqa_${DOMAIN}_train_SD.json"
RETAIN_TRAIN="${REPO}/data_new/clincqa_RD_only_200_c/clincqa_not_${JOINED}_train_RD.json"
SD_TEST="${REPO}/data_new/clincqa_SD_10/clincqa_${JOINED}_test_SD.json"
RD_TEST="${REPO}/data_new/clincqa_RD_10/clincqa_not_${JOINED}_test_RD.json"
UTILITY1="${REPO}/data_new/mrpcqa/mrpc_test.json"
UTILITY2="${REPO}/data_new/rteqa/rte_validation.json"

# ---- 체크포인트 ----
GA_CKPT="${GA_CKPT:-${BASELINE_DIR}/outputs/clincqa_fullft/ga/seed_${SEED}/t${STEP}_${DOMAIN}}"
GA_GD_CKPT="${GA_GD_CKPT:-${BASELINE_DIR}/outputs/clincqa_fullft/ga_gd/seed_${SEED}/t${STEP}_${DOMAIN}}"
NPO_CKPT="${NPO_CKPT:-${BASELINE_DIR}/outputs/clincqa_fullft/npo_kl/seed_${SEED}/t${STEP}_${DOMAIN}}"
# RULE = 논문 설정(MoE train bs 64) 재현 ckpt. outputs/clincqa_ckpt_repro 는 bs1024 라 쓰지 않는다.
RULE_CKPT="${RULE_CKPT:-${MODIFY_CODE}/outputs/repro_clinc_lunar_bs64/seed_${SEED}/t${STEP}_${DOMAIN}/lora_moe.pt}"

OUT_ROOT="${OUT_ROOT:-${HERE}/results/clinc_t${STEP}_bs64_${IMPL}}"
HYDRA_ROOT="${HYDRA_ROOT:-${HERE}/hydra_outputs}"
mkdir -p "${OUT_ROOT}"

# ---- SMOKE: 데이터를 앞 N개로 잘라 임시 파일로 쓴다 (속도 측정용) ----
if [[ -n "${SMOKE:-}" ]]; then
  SMOKE_DIR="${HERE}/.smoke_data"
  mkdir -p "${SMOKE_DIR}"
  for pair in "SD_TEST:${SD_TEST}" "RD_TEST:${RD_TEST}"; do
    var="${pair%%:*}"; src="${pair#*:}"
    dst="${SMOKE_DIR}/$(basename "${src}")"
    "${PY}" -c "
import json,sys
d=json.load(open(sys.argv[1]))[:int(sys.argv[3])]
json.dump(d,open(sys.argv[2],'w'),ensure_ascii=False)
print(f'[SMOKE] {sys.argv[2]} -> {len(d)} samples')
" "${src}" "${dst}" "${SMOKE}"
    printf -v "${var}" '%s' "${dst}"
  done
  OUT_ROOT="${OUT_ROOT}_smoke"; mkdir -p "${OUT_ROOT}"
fi

echo "[$(date)] quantization attack | CLINC t${STEP} (${DOMAIN})"
echo "  methods=${METHODS[*]}  modes=${MODES}  impl=${IMPL}  splits=${EVAL_SPLITS}  ckpt=${RULE_CKPT}"
echo "  forget test : ${SD_TEST}"
echo "  retain test : ${RD_TEST}"
echo "  -> ${OUT_ROOT}"

run_one () {   # $1=method  $2=mode
  local method="$1" mode="$2"
  local model_path="${BASE_MODEL}" moe=""
  case "${method}" in
    base)   ;;
    ga)     model_path="${GA_CKPT}"    ;;
    ga_gd)  model_path="${GA_GD_CKPT}" ;;
    npo_kl) model_path="${NPO_CKPT}"   ;;
    ours)   moe="${RULE_CKPT}"         ;;
    *) echo "[ERR] unknown method ${method}"; return 1 ;;
  esac

  if [[ "${method}" != "base" && "${method}" != "ours" && ! -d "${model_path}" ]]; then
    echo "[SKIP] ${method}: ckpt not found at ${model_path}"; return 0
  fi
  if [[ "${method}" == "ours" && ! -f "${moe}" ]]; then
    echo "[SKIP] ours: moe ckpt not found at ${moe}"; return 0
  fi

  if [[ "${method}" != "ours" && "${mode}" == "all" ]]; then
    echo "[SKIP] ${method}: adapter 없음 (all == backbone)"; return 0
  fi
  local tag="${method}_${mode}"
  [[ "${NBITS:-4}" != "4" ]] && tag="${tag}_${NBITS}bit"
  local out="${OUT_ROOT}/${tag}.json"
  if [[ -f "${out}" && -n "${SKIP_EXISTING:-}" ]]; then
    echo "[SKIP] ${tag}: already done"; return 0
  fi

  local extra=()
  [[ -n "${moe}" ]] && extra+=("+quant.moe_ckpt=${moe}")
  [[ -n "${MOE_BACKBONE_BF16:-}" ]] && extra+=("+quant.quantize_moe_backbone=false")

  echo "================ ${tag} ================"
  ( cd "${MODIFY_CODE}" && "${PY}" "${HERE}/eval_quant_clinc.py" \
      --config-name forget \
      data_name=clincqa \
      model_family="${MODEL_FAMILY}" \
      model_path="${model_path}" \
      seed="${SEED}" \
      task_id="${TASK_ID}" \
      layer_modified="${LAYER_MODIFIED}" \
      data_path="${FORGET_TRAIN}" \
      r_data_path="${RETAIN_TRAIN}" \
      use_unverified=false \
      use_harmful=true \
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
      +quant.nbits="${NBITS:-4}" \
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
"${PY}" "${HERE}/summarize_quant.py" "${OUT_ROOT}" || true
