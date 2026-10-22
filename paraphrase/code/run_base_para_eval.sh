#!/usr/bin/env bash
# unlearning 전 base 모델(O3_LLAMA31_{TOFU,ClincQA}, MoE 없음)로 원본/paraphrase SD·RD 전체 문항 1회 평가.
# base 는 step 마다 안 바뀌므로 마지막 step 파일(= 전체 문항)만 평가하고 step 별 점수는
# repro_check/base_para_steps.py 가 문항을 골라 다시 계산한다.
#   TOFU : ptofu_SD{suf}/tofu_step020_SD.json (800) + ptofu_RD{suf}/tofu_step020_RD.json (3200), eval bs 8
#   CLINC: clincqa_SD_10{suf}/clincqa_<9개>_test_SD.json (4050) + clincqa_RD_10{suf}/clincqa_not_<9개>_test_RD.json (450), eval bs 16
# 평가 경로 = quant_attack/eval_quant_{tofu,clinc}.py (mode=none, MoE 없음) = mmain 의 eval 과 같은 custom_evaluate*.
# 사용: CUDA_VISIBLE_DEVICES=1 bash repro_check/run_base_para_eval.sh tofu "orig para para_ctx"
set -uo pipefail
cd /home/hai_ssh/code/O3-LLM-UNLEARNING/ScienceQA_expriment/modify_code
DS="${1:?tofu|clinc}"; TYPES="${2:-orig para para_ctx para_struct para_bt para_noise}"
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
Q="$(cd ../quant_attack && pwd)"; D="$(cd ../data_new && pwd)"
OUT="$(pwd)/run_results/eval_basemoe_error/llama3.1-8b-instruct/para_eval/base"
LOG="$(pwd)/run_logs/para_eval/base"; mkdir -p "${OUT}" "${LOG}"
J9="work_travel_home_banking_credit_cards_kitchen_and_dining_auto_and_commute_utility_small_talk"
for ty in ${TYPES}; do
  suf=""; [[ "${ty}" != "orig" ]] && suf="_${ty}"
  out="${OUT}/${DS}_${ty}.json"
  [[ -s "${out}" ]] && { echo "[skip] ${out}"; continue; }
  echo "[$(date '+%F %T')] base ${DS} ${ty} -> ${out}"
  if [[ "${DS}" == tofu ]]; then
    "${PY}" "${Q}/eval_quant_tofu.py" --config-name forget \
      data_name=tofu model_family=llama3.1-8b-instruct \
      model_path=./train31Llama/trained_model/O3_LLAMA31_TOFU \
      seed=0 task_id="base_tofu_${ty}" layer_modified="[16]" eval_batch_size=8 \
      data_path="${D}/ptofu_SD_only/tofu_AG20_SD.json" r_data_path="${D}/ptofu_RD/tofu_step020_RD.json" \
      sampled_data_path="${D}/ptofu_SD${suf}/tofu_step020_SD.json" \
      retain_data_path="${D}/ptofu_RD${suf}/tofu_step020_RD.json" \
      utiliy_data_path1="${D}/tofu_real_authors/real_authors_test.json" \
      utiliy_data_path2="${D}/tofu_world_facts/world_facts_test.json" \
      load_unlearned_model=false save_unlearned_model=false save_path="${OUT}" \
      hydra.run.dir="${LOG}/hydra/${DS}_${ty}" \
      +quant.mode=none +quant.impl=fake +quant.tag="base_${DS}_${ty}" \
      "+quant.eval_splits=[forget,retained]" +quant.out_file="${out}" > "${LOG}/${DS}_${ty}.log" 2>&1
  else
    "${PY}" "${Q}/eval_quant_clinc.py" --config-name forget \
      data_name=clincqa model_family=llama3.1-8b-instruct \
      model_path=./train31Llama/trained_model/O3_LLAMA31_ClincQA \
      seed=0 task_id="base_clinc_${ty}" layer_modified="[16]" eval_batch_size=16 \
      data_path="${D}/clincqa_SD_only_200/clincqa_small_talk_train_SD.json" \
      r_data_path="${D}/clincqa_RD_only_200_c/clincqa_not_${J9}_train_RD.json" \
      use_unverified=false use_harmful=true \
      sampled_data_path="${D}/clincqa_SD_10${suf}/clincqa_${J9}_test_SD.json" \
      retain_data_path="${D}/clincqa_RD_10${suf}/clincqa_not_${J9}_test_RD.json" \
      utiliy_data_path1="${D}/mrpcqa/mrpc_test.json" utiliy_data_path2="${D}/rteqa/rte_validation.json" \
      load_unlearned_model=false save_unlearned_model=false save_path="${OUT}" \
      hydra.run.dir="${LOG}/hydra/${DS}_${ty}" \
      +quant.mode=none +quant.impl=fake +quant.tag="base_${DS}_${ty}" \
      "+quant.eval_splits=[forget,retained]" +quant.out_file="${out}" > "${LOG}/${DS}_${ty}.log" 2>&1
  fi
  rc=$?; [[ ${rc} -eq 0 && -s "${out}" ]] && echo "  done rc=0" || echo "  FAIL rc=${rc} (${LOG}/${DS}_${ty}.log)"
done
