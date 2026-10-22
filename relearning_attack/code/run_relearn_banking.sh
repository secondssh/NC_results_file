#!/usr/bin/env bash
# Relearning attack grid — RULE CLINC t8, 공격 domain = banking, GPU1.
#   target: base(unlearning 전) / backbone / moe(backbone+MoE)   data: in(CLINC banking 미사용 100) / out(BANKING77 1000)
#   epochs: 1(easy) / 5(hard)   + 공격 전 기준 base_none (RULE 공격 전 = repro_clinc_lunar_bs64 t8)
# 실행: setsid nohup bash relearn_attack/run_relearn_banking.sh > relearn_attack/logs/grid.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"; MC="${REPO}/modify_code"; D="${REPO}/data_new"
PY="${PY:-/home/hai_ssh/.conda/envs/lunar/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
J9="work_travel_home_banking_credit_cards_kitchen_and_dining_auto_and_commute_utility_small_talk"
CKPT="${MC}/outputs/repro_clinc_lunar_bs64/seed_0/t8_small_talk/lora_moe.pt"
OUT="${HERE}/results/banking_t8"; LOG="${HERE}/logs"; mkdir -p "${OUT}" "${LOG}"
TARGETS="${TARGETS:-base backbone moe}"; DATAS="${DATAS:-in out}"; EPOCHS="${EPOCHS_LIST:-1 5}"
declare -A DATA=([in]="${HERE}/data/in_banking_100.json" [out]="${HERE}/data/out_banking77_1000.json")

run() {  # $1 target $2 data-key $3 epochs
  local tag="$1_$2_ep$3"; [[ "$1" == none* ]] && tag="$1"
  local out="${OUT}/${tag}.json"
  [[ -s "${out}" ]] && { echo "[skip] ${tag}"; return; }
  local tgt="$1" mp="${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA" extra=()
  [[ "$1" == base_none ]] && tgt=none
  [[ "$1" == backbone || "$1" == moe ]] && extra+=("+attack.moe_ckpt=${CKPT}")
  echo "[$(date '+%F %T')] ${tag}"
  ( cd "${MC}" && "${PY}" "${HERE}/relearn_clinc.py" --config-name forget \
      data_name=clincqa model_family=llama3.1-8b-instruct model_path="${mp}" seed=0 \
      task_id=clincqa_t8_small_talk layer_modified="[16]" eval_batch_size=16 \
      data_path="${D}/clincqa_SD_only_200/clincqa_small_talk_train_SD.json" \
      r_data_path="${D}/clincqa_RD_only_200_c/clincqa_not_${J9}_train_RD.json" \
      use_unverified=false use_harmful=true \
      sampled_data_path="${D}/clincqa_SD_10/clincqa_${J9}_test_SD.json" \
      retain_data_path="${D}/clincqa_RD_10/clincqa_not_${J9}_test_RD.json" \
      utiliy_data_path1="${D}/mrpcqa/mrpc_test.json" utiliy_data_path2="${D}/rteqa/rte_validation.json" \
      load_unlearned_model=false save_unlearned_model=false \
      hydra.run.dir="${LOG}/hydra/${tag}" \
      +attack.target="${tgt}" +attack.data="${DATA[${2:-in}]}" +attack.epochs="${3:-1}" +attack.lr=1e-4 \
      +attack.out_file="${out}" "${extra[@]}" ) > "${LOG}/${tag}.log" 2>&1
  echo "   rc=$? $(grep '\[DONE\]' "${LOG}/${tag}.log" | cut -c1-160)"
}

# 학습이 들어간 조합을 먼저 (에러를 빨리 보기 위해), 순서: backbone → moe → base, 마지막에 공격 전 base 기준
for t in backbone moe base; do [[ " ${TARGETS} " == *" $t "* ]] || continue
  for d in ${DATAS}; do for e in ${EPOCHS}; do run "$t" "$d" "$e"; done; done; done
run base_none
echo "[$(date '+%F %T')] all done"
