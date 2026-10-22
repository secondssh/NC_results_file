#!/usr/bin/env bash
# Relearning attack 비교 — CLINC t0 (work), 공격 데이터 in_work_100 (unlearning 미사용 work 문장), 백본 공격 LoRA(attack_lora.py, 223 모듈)
#   GPU0: O3 (ep1, ep5)  — 첫 split 에서 학습+저장, 나머지 split 은 저장된 공격 LoRA 로드
#   GPU1: RULE (ep1, ep5) → GD (공격 전 bs16, ep1, ep5)
# 실행: setsid nohup bash relearn_attack/run_relearn_t0.sh > relearn_attack/logs/t0.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "${HERE}")"; MC="${REPO}/modify_code"; D="${REPO}/data_new"
PY=/home/hai_ssh/.conda/envs/lunar/bin/python
OUT="${HERE}/results/work_t0"; LOG="${HERE}/logs"; mkdir -p "${OUT}" "${LOG}"
DATA="${HERE}/data/in_work_100.json"
log() { echo "[$(date '+%F %T')] $*"; }

rule_gd() {  # $1 tag $2 target $3 model_path $4 epochs $5 moe_ckpt(or "")
  local out="${OUT}/$1.json" extra=()
  [[ -s "${out}" ]] && { log "[skip] $1"; return; }
  [[ -n "$5" ]] && extra+=("+attack.moe_ckpt=$5")
  log "$1"
  ( cd "${MC}" && CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY}" "${HERE}/relearn_clinc.py" --config-name forget \
      data_name=clincqa model_family=llama3.1-8b-instruct model_path="$3" seed=0 task_id=clincqa_t0_work \
      layer_modified="[16]" eval_batch_size=16 \
      data_path="${D}/clincqa_SD_only_200/clincqa_work_train_SD.json" \
      r_data_path="${D}/clincqa_RD_only_200_c/clincqa_not_work_train_RD.json" use_unverified=false use_harmful=true \
      sampled_data_path="${D}/clincqa_SD_10/clincqa_work_test_SD.json" \
      retain_data_path="${D}/clincqa_RD_10/clincqa_not_work_test_RD.json" \
      utiliy_data_path1="${D}/mrpcqa/mrpc_test.json" utiliy_data_path2="${D}/rteqa/rte_validation.json" \
      load_unlearned_model=false save_unlearned_model=false hydra.run.dir="${LOG}/hydra/t0_$1" \
      +attack.target="$2" +attack.impl=custom +attack.data="${DATA}" +attack.epochs="$4" +attack.lr=1e-4 \
      +attack.out_file="${out}" "${extra[@]}" ) > "${LOG}/t0_$1.log" 2>&1
  log "   rc=$? $(grep '\[DONE\]' "${LOG}/t0_$1.log" | sed 's#.*json | ##')"
}

o3() {  # $1 epochs
  local lora="${REPO}/unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_clinc9_origin/lora_force_random_work_force"
  local st="${OUT}/o3_ep$1_attack.pt" first=1
  declare -A T=([testSD]="${D}/clincqa_SD_10/clincqa_work_test_SD.json" [testRD]="${D}/clincqa_RD_10/clincqa_not_work_test_RD.json" \
                [mrpc]="${D}/mrpcqa/mrpc_test.json" [rte]="${D}/rteqa/rte_validation.json")
  for sp in testSD testRD mrpc rte; do
    local out="${OUT}/o3_ep$1_${sp}.json"; [[ -s "${out}" ]] && { first=0; continue; }
    local atk=(--attack_state "${st}"); [[ ! -s "${st}" ]] && atk=(--attack_train_data "${DATA}" --attack_epochs "$1" --attack_lr 1e-4 --attack_save "${st}")
    log "o3_ep$1 ${sp} (${atk[0]})"
    ( cd "${REPO}" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/hai_ssh/pylibs/peft0181 "${PY}" "${REPO}/quant_attack/eval_quant_o3_clinc.py" \
        --unlearn_topic "work_${sp}" --test_dataset "${T[$sp]}" \
        --base_model "${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA" --seed 0 --lora_weights "${lora}" \
        --ood_type work --ood_setting C --ood_setting_name clincqa \
        --ood_weights "${REPO}/ood_ocsvm/ood_checkpoints_clincqa_recent/ood_checkpoints_clincqa_0/" \
        --quant_mode none --result_file "${out}" "${atk[@]}" ) > "${LOG}/t0_o3_ep$1_${sp}.log" 2> "${LOG}/t0_o3_ep$1_${sp}.err"
    log "   rc=$? exact=$(python3 -c "import json;print(json.load(open('${out}'))['exact_acc'])" 2>/dev/null)"
  done
}

GD="${REPO}/unlearning_baseline_new/outputs/clincqa_fullft/ga_gd/seed_0/t0_work"
CB="${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA"
RCK="${MC}/outputs/repro_clinc_lunar_bs64/seed_0/t0_work/lora_moe.pt"
( o3 1; o3 5; log "GPU0 lane 끝" ) &
( rule_gd rule_ep1 backbone "${CB}" 1 "${RCK}"; rule_gd rule_ep5 backbone "${CB}" 5 "${RCK}"
  rule_gd gd_none none "${GD}" 1 ""; rule_gd gd_ep1 base "${GD}" 1 ""; rule_gd gd_ep5 base "${GD}" 5 ""; log "GPU1 lane 끝" ) &
wait; log "all done"
