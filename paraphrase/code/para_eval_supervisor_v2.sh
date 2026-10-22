#!/usr/bin/env bash
# Paraphrase 평가 감독 v2 (2026-10-20 교체): GPU 를 데이터셋별로 고정한다 (1분 간격)
#   GPU 0: TOFU 전용 2개
#   GPU 1: CLINC 전용 2개 — CLINC 가 다 끝나면 GPU1 에는 더 띄우지 않음
#          (GPU1 은 repro_check/run_gpu1_after_clinc.sh 가 router acc → base 모델 평가에 씀)
#   작업자(run_para_eval.sh)가 죽어서 결과 없이 남은 락은 풀어서 다시 돌게 함
#   145개(CLINC 45 + TOFU 100) 결과가 다 나오면 종료
# (modify_code에서) setsid nohup bash repro_check/para_eval_supervisor_v2.sh > run_logs/para_eval/supervisor_v2.log 2>&1 < /dev/null & disown
#   DRY=1 이면 한 번만 판단을 출력하고 아무것도 실행하지 않음
set -uo pipefail
MF="llama3.1-8b-instruct"
R="./run_results/eval_basemoe_error/${MF}/para_eval"
LOCKS="${R}/_locks"
TYPES="para para_ctx para_struct para_bt para_noise"
TOFU_DOMAINS=($(for i in $(seq 1 20); do printf "AG%02d " "$i"; done))
CLINC_DOMAINS=("work" "travel" "home" "banking" "credit_cards" "kitchen_and_dining" "auto_and_commute" "utility" "small_talk")
log() { echo "[$(date '+%F %T')] $*"; }

result_of() {  # $1=ds $2=type $3=t
  if [[ $1 == tofu ]]; then
    echo "${R}/tofu_$2/tofu_moe_fixed_layer16_0/lunar_${MF}_eval_base_tofu_t$3_${TOFU_DOMAINS[$3]}/forget_tofu_t$3_${TOFU_DOMAINS[$3]}.json"
  else
    echo "${R}/clinc_$2/clincqa_moe_fixed_layer16_0/lunar_${MF}_eval_base_clincqa_t$3_${CLINC_DOMAINS[$3]}/forget_clincqa_t$3_${CLINC_DOMAINS[$3]}.json"
  fi
}

running_jobs() {  # 지금 평가 중인 python의 "ds_type_tT" 목록
  for p in $(pgrep -f "mmain_(tofu_)?fixed.py"); do
    k=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | sed -nE 's#.*save_folder=para_eval/(tofu|clinc)_([a-z_]+)/.*_t([0-9]+)_[A-Za-z0-9_]+ ?.*#\1_\2_t\3#p')
    [[ -n "$k" ]] && echo "$k"
  done
}

scan() {  # 남은(결과 없음) 작업 중 바로 잡을 수 있는 수: pend_tofu / pend_clinc / total_done, 고아 락 정리
  local run; run=" $(running_jobs | tr '\n' ' ') "
  pend_tofu=0; pend_clinc=0; total_done=0
  for ty in ${TYPES}; do
    for ds in clinc tofu; do
      local last=19; [[ $ds == clinc ]] && last=8
      for t in $(seq 0 ${last}); do
        if [[ -s "$(result_of $ds $ty $t)" ]]; then total_done=$((total_done + 1)); continue; fi
        local key="${ds}_${ty}_t${t}"
        if [[ -d "${LOCKS}/${key}" ]]; then
          [[ "${run}" == *" ${key} "* ]] && continue
          log "고아 락 해제: ${key} (실행 중인 평가 없음, 결과 없음)"
          [[ -z "${DRY:-}" ]] && rmdir "${LOCKS}/${key}"
        fi
        [[ $ds == tofu ]] && pend_tofu=$((pend_tofu + 1)) || pend_clinc=$((pend_clinc + 1))
      done
    done
  done
}

workers_on() {  # $1=gpu → 그 GPU에서 도는 run_para_eval.sh 작업자 수
  local n=0
  for p in $(pgrep -f "bash repro_check/run_para_eval.sh"); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && n=$((n + 1))
  done
  echo $n
}

launch() {  # $1=gpu $2=datasets
  local tag; tag="$(date +%m%d_%H%M%S)"
  log "작업자 실행: GPU $1, DATASETS=$2"
  [[ -n "${DRY:-}" ]] && return
  CUDA_VISIBLE_DEVICES=$1 DATASETS="$2" setsid nohup bash repro_check/run_para_eval.sh \
    > "run_logs/para_eval/worker_gpu$1_${2// /_}_${tag}.log" 2>&1 < /dev/null &
  sleep 5
}

log "supervisor start (pid $$)"
while :; do
  scan
  if (( total_done >= 145 )); then log "145개 전부 완료 → 종료"; break; fi
  for gpu in 0 1; do
    want=2
    have=$(workers_on $gpu)
    while (( have < want )); do
      if [[ $gpu == 1 ]]; then
        (( pend_clinc > 0 )) && ds=clinc || break
      else
        (( pend_tofu > 0 )) && ds=tofu || break
      fi
      launch $gpu $ds
      have=$((have + 1))
      [[ $ds == tofu ]] && pend_tofu=$((pend_tofu - 1)) || pend_clinc=$((pend_clinc - 1))
    done
  done
  [[ -n "${DRY:-}" ]] && { log "DRY: done=${total_done} pend_tofu=${pend_tofu} pend_clinc=${pend_clinc} gpu0=$(workers_on 0) gpu1=$(workers_on 1)"; break; }
  sleep 60
done
