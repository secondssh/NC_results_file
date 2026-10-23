#!/usr/bin/env bash
# O3 LLM LoRA 학습 시간 / 메모리 (CLINC t0, 원래 train_unlearn_lora_clinc.sh t0 인자 그대로), o3_lora_train_wrapper.py 로 실행
# (chat_template.jinja 보충만, O3 추론 측정과 같은 방식). run_gd_tofu_infer.sh 끝난 뒤 ("gd tofu finished"), GPU 2장 90초 연속 idle 일 때 GPU0.
#   lora_time_w : monitor --light, 20 epoch (원래 설정 전부)   lora_mem_w : monitor PSS, 1 epoch
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MC="$(dirname "${HERE}")"; REPO="$(dirname "${MC}")"
LP=/home/hai_ssh/.conda/envs/lunar/bin/python
R="${HERE}/results/v2/o3_train"; L="${HERE}/logs/v2/o3_train"; mkdir -p "${R}" "${L}"
SCR=/tmp/claude-10011/-home-hai-ssh/1484149c-2ae3-43d6-be42-03577263b907/scratchpad/o3_train
log() { echo "[$(date '+%F %T')] $*"; }
wait_idle() { local ok=0; while (( ok < 4 )); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then ok=$((ok + 1)); else ok=0; fi
  (( ok < 4 )) && sleep 30; done; }
until grep -q "gd tofu finished" "${HERE}/logs/v2/gd_tofu.log" 2>/dev/null; do sleep 60; done
for job in "lora_time_w|--light --interval 2|20" "lora_mem_w||1"; do
  IFS='|' read -r name mopt ep <<< "${job}"
  [[ -s "${R}/mon_${name}.json" ]] && { log "[skip] ${name}"; continue; }
  wd="${SCR}/${name}"; wait_idle; rm -rf "${wd}"; mkdir -p "${wd}"; log "START ${name}"
  ( cd "${wd}" && env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=/home/hai_ssh/pylibs/peft0181:/home/hai_ssh/pylibs/fire \
      "${LP}" "${HERE}/monitor.py" ${mopt} --out "${R}/mon_${name}.json" -- \
      "${LP}" "${HERE}/o3_lora_train_wrapper.py" "${REPO}/train_unlearn_lora_o.py" \
      --base_model "${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA" \
      --data_path "${REPO}/data_new/clincqa_random_force/clincqa_work_train_random_force.json" --output_dir "${wd}/out" --seed 0 \
      --batch_size 128 --micro_batch_size 8 --num_epochs "${ep}" --learning_rate 3e-4 --cutoff_len 256 --val_set_size 1 \
      --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --lora_weights --ood_weight 1,1 --orthogonal_loss_weight 0.1 \
      --train_on_inputs --group_by_length --add_eos_token --resume_from_checkpoint --lora_start_layer 0 ) > "${L}/${name}.log" 2>&1
  log "END ${name} rc=$? | $(grep -E '^\[monitor\]' "${L}/${name}.log" | tail -1) | $(grep -ao "'train_runtime': [0-9.]*" "${L}/${name}.log" | tail -1)"
  rm -rf "${wd}/out"
done
log "o3 lora finished"
