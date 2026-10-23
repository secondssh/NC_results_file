#!/usr/bin/env bash
# Relearning attack — CLINC t3 (banking), out: BANKING77 1000, full-FT baseline (GA, NPO+KL, GD)
#   각 방법: 공격 전 (target=none) → ep1 → ep5. 공격 = 백본 LoRA (attack_lora.py, impl=custom, 223 모듈), lr 1e-4
#   (full-FT 라 별도 unlearning 모듈 없음 → "백본 LoRA" 한 조건). 평가 eval bs 16, RULE/O3 t3 와 같은 문항
#   GPU0: GA → GD 공격 전, ep1     GPU1: NPO+KL → GD ep5
# 실행: setsid nohup bash relearn_attack/run_relearn_t3_baselines.sh > relearn_attack/logs/t3_baselines.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "${HERE}")"; MC="${REPO}/modify_code"; D="${REPO}/data_new"
PY=/home/hai_ssh/.conda/envs/lunar/bin/python
J="work_travel_home_banking"
OUT="${HERE}/results/banking_t3"; LOG="${HERE}/logs/t3_baselines"; mkdir -p "${OUT}" "${LOG}"
DATA="${HERE}/data/out_banking77_1000.json"
CK="${REPO}/unlearning_baseline_new/outputs/clincqa_fullft"
log() { echo "[$(date '+%F %T')] $*"; }

run() {  # $1 gpu $2 method(ga|npo_kl|ga_gd) $3 none|1|5
  local tag="$2_out_ep$3" tgt=base ep="$3"
  [[ "$3" == none ]] && { tag="$2_none"; tgt=none; ep=1; }
  local out="${OUT}/${tag}.json"
  [[ -s "${out}" ]] && { log "[skip] ${tag}"; return; }
  log "GPU$1 ${tag}"
  ( cd "${MC}" && CUDA_VISIBLE_DEVICES=$1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "${PY}" "${HERE}/relearn_clinc.py" --config-name forget \
      data_name=clincqa model_family=llama3.1-8b-instruct model_path="${CK}/$2/seed_0/t3_banking" seed=0 \
      task_id=clincqa_t3_banking layer_modified="[16]" eval_batch_size=16 \
      data_path="${D}/clincqa_SD_only_200/clincqa_banking_train_SD.json" \
      r_data_path="${D}/clincqa_RD_only_200_c/clincqa_not_${J}_train_RD.json" use_unverified=false use_harmful=true \
      sampled_data_path="${D}/clincqa_SD_10/clincqa_${J}_test_SD.json" \
      retain_data_path="${D}/clincqa_RD_10/clincqa_not_${J}_test_RD.json" \
      utiliy_data_path1="${D}/mrpcqa/mrpc_test.json" utiliy_data_path2="${D}/rteqa/rte_validation.json" \
      load_unlearned_model=false save_unlearned_model=false hydra.run.dir="${LOG}/hydra/${tag}" \
      +attack.target="${tgt}" +attack.impl=custom +attack.data="${DATA}" +attack.epochs="${ep}" +attack.lr=1e-4 \
      +attack.out_file="${out}" ) > "${LOG}/${tag}.log" 2>&1
  log "   GPU$1 ${tag} rc=$? $(grep '\[DONE\]' "${LOG}/${tag}.log" | sed 's#.*json | ##')"
}
( for e in none 1 5; do run 0 ga $e; done; run 0 ga_gd none; run 0 ga_gd 1; log "GPU0 lane 끝" ) &
( for e in none 1 5; do run 1 npo_kl $e; done; run 1 ga_gd 5; log "GPU1 lane 끝" ) &
wait; log "all done"
cd "${MC}/readme_main_results/_build" && "${PY}" build_all.py relearn && log "relearn 표 재생성 끝"
