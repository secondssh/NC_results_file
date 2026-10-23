#!/usr/bin/env bash
# O3 학습 시간 / 메모리 (CLINC t0 = work, seed 0), 원래 학습 명령 그대로 (train_ood.sh / train_unlearn_lora_clinc.sh 의 t0 인자).
#   detector : train_ood.py, o3 env (detector 저장 peft 0.10.0) + PYTHONPATH openpyxl (엑셀 로그용, 연산 무관)
#              데이터 = data/clincqa_SD_only_10 (work 1500 → 80% 학습 = 저장 feature 1200 과 일치), data/clincqa_RD_only_10/clincqa_not_work (13500)
#   LLM LoRA : train_unlearn_lora_o.py, lunar + PYTHONPATH peft0181 (LLM LoRA 저장 peft 0.18.1) + fire (CLI, 연산 무관)
#              data_new/clincqa_random_force/clincqa_work_train_random_force.json (1500), bs 128 / micro 8, 20 epoch, lr 3e-4, cutoff 256
# 시간 = monitor --light (원래 epoch 수 전부), host 메모리 = monitor PSS 별도 실행 (LoRA 는 1 epoch). 산출물은 scratch 에만 (기존 ckpt 안 건드림).
# run_v2_timing.sh 가 끝난 뒤 ("timing finished"), GPU 2장이 90초 연속 비면 GPU0 에서 시작.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MC="$(dirname "${HERE}")"; REPO="$(dirname "${MC}")"
LP=/home/hai_ssh/.conda/envs/lunar/bin/python; OP=/home/hai_ssh/miniconda3/envs/o3/bin/python
R="${HERE}/results/v2/o3_train"; L="${HERE}/logs/v2/o3_train"; mkdir -p "${R}" "${L}"
SCR=/tmp/claude-10011/-home-hai-ssh/1484149c-2ae3-43d6-be42-03577263b907/scratchpad/o3_train
log() { echo "[$(date '+%F %T')] $*"; }
wait_idle() { local ok=0; while (( ok < 4 )); do
  if [[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then ok=$((ok + 1)); else ok=0; fi
  (( ok < 4 )) && sleep 30; done; }
until grep -q "timing finished" "${HERE}/logs/v2/timing.log" 2>/dev/null; do sleep 60; done
OOD_ARGS=(--unlearn_dataset clincqa_work --ood_dataset ood_clincqa --base_unlearn_path "${REPO}/data/clincqa_SD_only_10"
          --base_ood_path "${REPO}/data/clincqa_RD_only_10/clincqa_not_work" --seed 0)
lora_args() {  # $1 = output dir, $2 = epochs
  echo --base_model "${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA" \
    --data_path "${REPO}/data_new/clincqa_random_force/clincqa_work_train_random_force.json" --output_dir "$1" --seed 0 \
    --batch_size 128 --micro_batch_size 8 --num_epochs "$2" --learning_rate 3e-4 --cutoff_len 256 --val_set_size 1 \
    --lora_r 8 --lora_alpha 16 --lora_dropout 0.05 --lora_weights --ood_weight 1,1 --orthogonal_loss_weight 0.1 \
    --train_on_inputs --group_by_length --add_eos_token --resume_from_checkpoint --lora_start_layer 0
}
run() {  # $1 이름, $2 monitor 옵션, $3 env 접두, $4 작업 폴더, 나머지 = 명령
  local name=$1 mopt=$2 envs=$3 wd=$4; shift 4
  [[ -s "${R}/mon_${name}.json" ]] && { log "[skip] ${name}"; return; }
  wait_idle; rm -rf "${wd}"; mkdir -p "${wd}"; log "START ${name}"
  ( cd "${wd}" && env CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 ${envs} "${LP}" "${HERE}/monitor.py" ${mopt} --out "${R}/mon_${name}.json" -- "$@" ) > "${L}/${name}.log" 2>&1
  log "END ${name} rc=$? | $(grep -E '^\[monitor\]' "${L}/${name}.log" | tail -1) | $(grep -ao "'train_runtime': [0-9.]*" "${L}/${name}.log" | tail -1)"
}
run ood_time "--light --interval 2" "PYTHONPATH=/home/hai_ssh/pylibs/openpyxl" "${SCR}/ood_time" "${OP}" "${REPO}/train_ood.py" "${OOD_ARGS[@]}"
run lora_time "--light --interval 2" "PYTHONPATH=/home/hai_ssh/pylibs/peft0181:/home/hai_ssh/pylibs/fire" "${SCR}/lora_time" \
  "${LP}" "${REPO}/train_unlearn_lora_o.py" $(lora_args "${SCR}/lora_time/out" 20)
run ood_mem "" "PYTHONPATH=/home/hai_ssh/pylibs/openpyxl" "${SCR}/ood_mem" "${OP}" "${REPO}/train_ood.py" "${OOD_ARGS[@]}"
run lora_mem "" "PYTHONPATH=/home/hai_ssh/pylibs/peft0181:/home/hai_ssh/pylibs/fire" "${SCR}/lora_mem" \
  "${LP}" "${REPO}/train_unlearn_lora_o.py" $(lora_args "${SCR}/lora_mem/out" 1)
log "o3 train finished"
