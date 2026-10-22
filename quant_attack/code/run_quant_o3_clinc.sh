#!/usr/bin/env bash
# Quantization attack on O3 (baseline) — CLINC t8 (9개 도메인 누적 LoRA, seed 0).
#
#   표:  bf16(원래, O3 결과 json) | backbone 4bit | backbone + LoRA 4bit
#   평가 split / 인자 / OOD 경로 규칙은 ../eval_soft_infer_clinc.sh 의 마지막 도메인(small_talk)과 같다.
#   (trainSD 는 제외)
#
# 사용법 (ScienceQA_expriment 기준 어디서든)
#   CUDA_VISIBLE_DEVICES=1 bash quant_attack/run_quant_o3_clinc.sh
#   MODES="none backbone all"  SPLITS="testSD testRD mrpc rte"
#   OOD_DET=recent|9           # ood_ocsvm/ood_checkpoints_clincqa_{recent,9} 중 무엇이 원래 결과인지 미확정
#   MAX_SAMPLES=50             # 앞 50개만 (원래 결과와 비교: quant_attack/check_o3_clinc_repro.py)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"                       # ScienceQA_expriment
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
# O3 CLINC adapter 는 peft 0.18.1 로 저장됨 (lunar 의 0.19.1 과 dispatch 시그니처 비호환) → 별도 폴더의 0.18.1
export PYTHONPATH="/home/hai_ssh/pylibs/peft0181${PYTHONPATH:+:${PYTHONPATH}}"
SEED="${SEED:-0}"
MODES="${MODES:-backbone all}"
SPLITS="${SPLITS:-testSD testRD mrpc rte}"
OOD_DET="${OOD_DET:-recent}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

DOMAINS=(work travel home banking credit_cards kitchen_and_dining auto_and_commute utility small_talk)
STEP="${STEP:-8}"
UNLEAN_D="${DOMAINS[$STEP]}"
TYPE="$(IFS=_; echo "${DOMAINS[*]:0:$((STEP + 1))}")"
LORA="${REPO}/unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_${SEED}_o_unlearn_lora_force_checkpoints_clinc9_origin/lora_force_random"
for d in "${DOMAINS[@]:0:$((STEP + 1))}"; do LORA+="_${d}_force"; done
BASE_MODEL="${BASE_MODEL:-${REPO}/modify_code/train31Llama/trained_model/O3_LLAMA31_ClincQA}"
OOD_WEIGHTS="${REPO}/ood_ocsvm/ood_checkpoints_clincqa_${OOD_DET}/ood_checkpoints_clincqa_${SEED}/"

OUT_ROOT="${OUT_ROOT:-${HERE}/results/o3_clinc_t${STEP}_ood${OOD_DET}}"
[[ "${MAX_SAMPLES}" != "0" ]] && OUT_ROOT="${OUT_ROOT}_n${MAX_SAMPLES}"
mkdir -p "${OUT_ROOT}"
[[ -d "${LORA}" ]] || { echo "[ERR] LoRA 없음: ${LORA}"; exit 1; }

declare -A TEST=(
  [testSD]="${REPO}/data_new/clincqa_SD_10/clincqa_${TYPE}_test_SD.json"
  [testRD]="${REPO}/data_new/clincqa_RD_10/clincqa_not_${TYPE}_test_RD.json"
  [mrpc]="${REPO}/data_new/mrpcqa/mrpc_test.json"
  [rte]="${REPO}/data_new/rteqa/rte_validation.json"
)

echo "[$(date)] O3 quantization attack | CLINC t${STEP} (${UNLEAN_D}) seed ${SEED}"
echo "  lora=${LORA}"
echo "  ood=${OOD_WEIGHTS}  modes=${MODES}  splits=${SPLITS}  max_samples=${MAX_SAMPLES}"
echo "  -> ${OUT_ROOT}"

for mode in ${MODES}; do
  for sp in ${SPLITS}; do
    out="${OUT_ROOT}/${mode}_${sp}.json"
    if [[ -f "${out}" && -n "${SKIP_EXISTING:-}" ]]; then echo "[SKIP] ${out}"; continue; fi
    echo "================ ${mode} / ${sp} ================"
    ( cd "${REPO}" && "${PY}" "${HERE}/eval_quant_o3_clinc.py" \
        --unlearn_topic "${UNLEAN_D}_${sp}" \
        --test_dataset "${TEST[$sp]}" \
        --base_model "${BASE_MODEL}" \
        --seed "${SEED}" \
        --lora_weights "${LORA}" \
        --ood_type "${UNLEAN_D}" \
        --ood_setting C \
        --ood_setting_name clincqa \
        --ood_weights "${OOD_WEIGHTS}" \
        --quant_mode "${mode}" \
        --result_file "${out}" \
        --max_samples "${MAX_SAMPLES}" ) 2> "${out%.json}.stderr.log"
  done
done
echo "[$(date)] done -> ${OUT_ROOT}"
