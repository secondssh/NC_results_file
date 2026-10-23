#!/usr/bin/env bash
# Relearning attack 보강 — CLINC t0 (work) out-domain: 공격 데이터 out_massive_work_1000 (MASSIVE 업무 비서 계열 8 intent)
#   GPU0: O3 백본 LoRA (O3 LoRA 고정) ep1, ep5 → O3 백본 LoRA + O3 LoRA ep1, ep5
#   GPU1: RULE 백본 LoRA (MoE 고정) ep1, ep5 → RULE 백본 LoRA + MoE ep1, ep5 → GD 백본 LoRA ep1, ep5
#   공격 LoRA = attack_lora.py (impl=custom, 223 모듈), lr 1e-4 — run_relearn_t0.sh (실험 B) 와 같은 설정, 데이터만 다름
# 실행: setsid nohup bash relearn_attack/run_relearn_t0_out.sh > relearn_attack/logs/t0_out.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "${HERE}")"; MC="${REPO}/modify_code"; D="${REPO}/data_new"
PY=/home/hai_ssh/.conda/envs/lunar/bin/python
OUT="${HERE}/results/work_t0_out"; LOG="${HERE}/logs/t0_out"; mkdir -p "${OUT}" "${LOG}"
DATA="${HERE}/data/out_massive_work_1000.json"
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
      load_unlearned_model=false save_unlearned_model=false hydra.run.dir="${LOG}/hydra/$1" \
      +attack.target="$2" +attack.impl=custom +attack.data="${DATA}" +attack.epochs="$4" +attack.lr=1e-4 \
      +attack.out_file="${out}" "${extra[@]}" ) > "${LOG}/$1.log" 2>&1
  log "   rc=$? $(grep '\[DONE\]' "${LOG}/$1.log" | sed 's#.*json | ##')"
}

o3() {  # $1 tag(o3_backbone|o3_lora) $2 epochs $3 extra flag
  local lora="${REPO}/unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_clinc9_origin/lora_force_random_work_force"
  local tag="$1_ep$2" st="${OUT}/$1_ep$2_attack.pt"
  declare -A T=([testSD]="${D}/clincqa_SD_10/clincqa_work_test_SD.json" [testRD]="${D}/clincqa_RD_10/clincqa_not_work_test_RD.json" \
                [mrpc]="${D}/mrpcqa/mrpc_test.json" [rte]="${D}/rteqa/rte_validation.json")
  for sp in testSD testRD mrpc rte; do
    local out="${OUT}/${tag}_${sp}.json"; [[ -s "${out}" ]] && continue
    local atk=(--attack_state "${st}")
    [[ ! -s "${st}" ]] && atk=(--attack_train_data "${DATA}" --attack_epochs "$2" --attack_lr 1e-4 --attack_save "${st}" ${3:+$3})
    log "${tag} ${sp} (${atk[0]})"
    ( cd "${REPO}" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/hai_ssh/pylibs/peft0181 "${PY}" "${REPO}/quant_attack/eval_quant_o3_clinc.py" \
        --unlearn_topic "work_${sp}" --test_dataset "${T[$sp]}" \
        --base_model "${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA" --seed 0 --lora_weights "${lora}" \
        --ood_type work --ood_setting C --ood_setting_name clincqa \
        --ood_weights "${REPO}/ood_ocsvm/ood_checkpoints_clincqa_recent/ood_checkpoints_clincqa_0/" \
        --quant_mode none --result_file "${out}" "${atk[@]}" ) > "${LOG}/${tag}_${sp}.log" 2> "${LOG}/${tag}_${sp}.err"
    log "   rc=$? exact=$("${PY}" -c "import json;print(round(json.load(open('${out}'))['exact_acc'],2))" 2>/dev/null)"
  done
}

GD="${REPO}/unlearning_baseline_new/outputs/clincqa_fullft/ga_gd/seed_0/t0_work"
CB="${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA"
RCK="${MC}/outputs/repro_clinc_lunar_bs64/seed_0/t0_work/lora_moe.pt"
( o3 o3_backbone 1 ""; o3 o3_backbone 5 ""; o3 o3_lora 1 "--attack_train_o3_lora"; o3 o3_lora 5 "--attack_train_o3_lora"
  log "GPU0 lane 끝" ) &
( rule_gd rule_backbone_ep1 backbone "${CB}" 1 "${RCK}"; rule_gd rule_backbone_ep5 backbone "${CB}" 5 "${RCK}"
  rule_gd rule_moe_ep1 moe "${CB}" 1 "${RCK}"; rule_gd rule_moe_ep5 moe "${CB}" 5 "${RCK}"
  rule_gd gd_ep1 base "${GD}" 1 ""; rule_gd gd_ep5 base "${GD}" 5 ""; log "GPU1 lane 끝" ) &
wait; log "all done"
