#!/usr/bin/env bash
# Cross domain / cross author 입력 평가: CLINC t4,t8 / TOFU t10,t19, seed0, F+R 와 R+F.  (README "Cross Domain or Cross Author Input Evaluation")
#
# 평가 경로 = quant_attack/eval_quant_{clinc,tofu}.py 의 quant.mode=none (MoE 없이 bf16, 원래 eval 과 생성문 일치 확인됨:
#   quant_attack/logs/{clinc,tofu}_smoke.log REPRO_OK). 평가 데이터만 multi_attack/data/{ds}_{FR,RF}.json 로 바꾸고
#   "forget" split 하나만 돌린다 (주 필드 = F 쪽 → 저장 choice_accuracy = F 누출). eval batch CLINC 16 / TOFU 8, 생성 설정 동일.
#   method: ours = RULE seed0 그 시점 ckpt, base = unlearning 전 모델 (MoE 없음, 섞인 입력 자체의 난이도 기준선)
# 순서: (데이터 없으면 생성) → eval (결과 있으면 건너뜀) → RULE router(teacher forcing) → 채점표
#
# 사용 (modify_code 에서):
#   CUDA_VISIBLE_DEVICES=1 setsid nohup bash multi_attack/run_cross_eval.sh > multi_attack/logs/run.log 2>&1 < /dev/null & disown
#   DATASETS="clinc" | "tofu" | "clinc tofu"(기본)   METHODS="ours base"(기본)   ORDERS="FR RF"(기본)
#   CLINC_STEPS="4 8"(기본)  TOFU_STEPS="10 19"(기본)   (데이터는 make_cross_data.py STEPS 에 있는 시점만)
#   SMOKE=16 ...   # 각 세트 앞 16개만 (파이프라인 점검용, 결과는 results_smoke/)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODIFY_CODE="$(dirname "${HERE}")"
REPO="$(dirname "${MODIFY_CODE}")"
Q="${REPO}/quant_attack"
D="${REPO}/data_new"
cd "${MODIFY_CODE}"

PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
DATASETS="${DATASETS:-clinc tofu}"
METHODS="${METHODS:-ours base}"
ORDERS="${ORDERS:-FR RF}"
CLINC_STEPS="${CLINC_STEPS:-4 8}"
TOFU_STEPS="${TOFU_STEPS:-10 19}"
MODEL_FAMILY="llama3.1-8b-instruct"
CLINC_DOMAINS=("work" "travel" "home" "banking" "credit_cards" "kitchen_and_dining" "auto_and_commute" "utility" "small_talk")

DATA_DIR="${HERE}/data"
OUT="${HERE}/results"; LOG="${HERE}/logs"
if [[ -n "${SMOKE:-}" ]]; then OUT="${HERE}/results_smoke"; LOG="${HERE}/logs/smoke"; fi
mkdir -p "${OUT}" "${LOG}"

log() { echo "[$(date '+%F %T')] $*"; }

[[ -s "${DATA_DIR}/manifest.json" ]] || "${PY}" "${HERE}/make_cross_data.py" || exit 1

data_file() {  # $1=ds $2=t $3=order → 평가 파일 경로 (SMOKE 면 앞 N개 잘라 만든 복사본)
  local name="$1_t$2_$3"
  local src="${DATA_DIR}/${name}.json"
  if [[ -n "${SMOKE:-}" ]]; then
    local dst="${DATA_DIR}/smoke/${name}.json"; mkdir -p "${DATA_DIR}/smoke"
    "${PY}" -c "import json,sys; json.dump(json.load(open(sys.argv[1]))[:int(sys.argv[3])],open(sys.argv[2],'w'),ensure_ascii=False)" \
      "${src}" "${dst}" "${SMOKE}"
    src="${dst}"
  fi
  echo "${src}"
}

run_eval() {  # $1=ds $2=t $3=method $4=order
  local ds=$1 t=$2 method=$3 order=$4 data out tag ck dom joined step extra=()
  [[ -s "${DATA_DIR}/${ds}_t${t}_${order}.json" ]] || { log "[FAIL] 데이터 없음 ${ds}_t${t}_${order} (make_cross_data.py STEPS)"; return 1; }
  data="$(data_file "${ds}" "${t}" "${order}")"
  tag="${ds}_t${t}_${method}_${order}"; out="${OUT}/${tag}.json"
  [[ -s "${out}" ]] && { log "[skip] ${tag}"; return 0; }
  if [[ "${ds}" == clinc ]]; then
    dom="${CLINC_DOMAINS[$t]}"; joined="$(IFS=_; echo "${CLINC_DOMAINS[*]:0:$((t + 1))}")"
    ck="${MODIFY_CODE}/outputs/repro_clinc_lunar_bs64/seed_0/t${t}_${dom}/lora_moe.pt"
  else
    dom="$(printf "AG%02d" $((t + 1)))"; step="$(printf "%03d" $((t + 1)))"
    ck="${MODIFY_CODE}/outputs/tofu_ckpt_repro_lunar/seed_0/t${t}_${dom}/lora_moe.pt"
  fi
  if [[ "${method}" == ours ]]; then
    [[ -f "${ck}" ]] || { log "[FAIL] ckpt 없음 ${ck}"; return 1; }
    extra+=("+quant.moe_ckpt=${ck}")
  fi
  log "EVAL ${tag} <- ${data}"
  local start=$(date +%s)
  if [[ "${ds}" == tofu ]]; then
    "${PY}" "${Q}/eval_quant_tofu.py" --config-name forget \
      data_name=tofu model_family="${MODEL_FAMILY}" \
      model_path=./train31Llama/trained_model/O3_LLAMA31_TOFU \
      seed=0 task_id="cross_${tag}" layer_modified="[16]" eval_batch_size=8 \
      data_path="${D}/ptofu_SD_only/tofu_${dom}_SD.json" r_data_path="${D}/ptofu_RD/tofu_step${step}_RD.json" \
      sampled_data_path="${data}" retain_data_path="${D}/ptofu_RD/tofu_step${step}_RD.json" \
      utiliy_data_path1="${D}/tofu_real_authors/real_authors_test.json" \
      utiliy_data_path2="${D}/tofu_world_facts/world_facts_test.json" \
      load_unlearned_model=false save_unlearned_model=false save_path="${OUT}" \
      hydra.run.dir="${LOG}/hydra/${tag}" \
      +quant.mode=none +quant.impl=fake +quant.tag="${tag}" +quant.router_stats=false \
      "+quant.eval_splits=[forget]" +quant.out_file="${out}" "${extra[@]}" > "${LOG}/${tag}.log" 2>&1
  else
    "${PY}" "${Q}/eval_quant_clinc.py" --config-name forget \
      data_name=clincqa model_family="${MODEL_FAMILY}" \
      model_path=./train31Llama/trained_model/O3_LLAMA31_ClincQA \
      seed=0 task_id="cross_${tag}" layer_modified="[16]" eval_batch_size=16 \
      data_path="${D}/clincqa_SD_only_200/clincqa_${dom}_train_SD.json" \
      r_data_path="${D}/clincqa_RD_only_200_c/clincqa_not_${joined}_train_RD.json" \
      use_unverified=false use_harmful=true \
      sampled_data_path="${data}" retain_data_path="${D}/clincqa_RD_10/clincqa_not_${joined}_test_RD.json" \
      utiliy_data_path1="${D}/mrpcqa/mrpc_test.json" utiliy_data_path2="${D}/rteqa/rte_validation.json" \
      load_unlearned_model=false save_unlearned_model=false save_path="${OUT}" \
      hydra.run.dir="${LOG}/hydra/${tag}" \
      +quant.mode=none +quant.impl=fake +quant.tag="${tag}" +quant.router_stats=false \
      "+quant.eval_splits=[forget]" +quant.out_file="${out}" "${extra[@]}" > "${LOG}/${tag}.log" 2>&1
  fi
  local rc=$?
  if [[ ${rc} -eq 0 && -s "${out}" ]]; then log "DONE ${tag} ($(( ($(date +%s) - start) / 60 )) min)"
  else log "FAIL ${tag} rc=${rc} (${LOG}/${tag}.log)"; rm -f "${out}"; return 1; fi
}

fail=0
steps_of() { [[ "$1" == clinc ]] && echo "${CLINC_STEPS}" || echo "${TOFU_STEPS}"; }
# 순서: 시점 오름차순 (같은 시점 안에서 ours → base, FR → RF)
for ds in ${DATASETS}; do
  for t in $(steps_of "${ds}"); do
    for method in ${METHODS}; do
      for order in ${ORDERS}; do run_eval "${ds}" "${t}" "${method}" "${order}" || fail=$((fail + 1)); done
    done
  done
done

if [[ " ${METHODS} " == *" ours "* ]]; then
  for ds in ${DATASETS}; do
    log "ROUTER ${ds} t=[$(steps_of "${ds}")] (RULE, teacher forcing on saved generations)"
    "${PY}" "${HERE}/router_cross.py" --ds "${ds}" --steps $(steps_of "${ds}") --orders ${ORDERS} --res_dir "${OUT}" \
      > "${LOG}/router_${ds}.log" 2>&1 || { log "FAIL router ${ds} (${LOG}/router_${ds}.log)"; fail=$((fail + 1)); }
  done
fi

log "SCORE"
"${PY}" "${HERE}/score_cross.py" --res_dir "${OUT}" --datasets ${DATASETS} > "${LOG}/score.log" 2>&1 \
  && cat "${OUT}/cross_table.md" || { log "FAIL score (${LOG}/score.log)"; fail=$((fail + 1)); }
log "finished (fail=${fail})"
