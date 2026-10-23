#!/usr/bin/env bash
# Relearning attack — CLINC t3 (banking 을 지운 시점), 공격 domain = banking, GPU1 순서 실행.
#   1) O3 t3 smoke: 공격 없이 testSD 50 문항 → 원래 O3 결과(test_seed0_banking_testSD.json) 와 pred 일치 90% 미만이면 중단
#   2) RULE t3: backbone / moe × in / out × ep1 / ep5 (8 run, relearn_clinc.py, t8 grid 와 같은 설정)
#   3) O3 t3: 백본 공격 LoRA 만 (O3 LoRA 고정) × in / out × ep1 / ep5 (4 run)
#   4) O3 t3: 백본 공격 LoRA + O3 unlearning LoRA 같이 학습 (--attack_train_o3_lora) (4 run)
#   base 는 ckpt 무관 → t8 base 공격 결과를 t3 SD/RD 로 재분할 (GPU 안 씀, readme_main_results/_build 에서)
# 실행: setsid nohup bash relearn_attack/run_relearn_t3.sh > relearn_attack/logs/t3.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "${HERE}")"; MC="${REPO}/modify_code"; D="${REPO}/data_new"
PY=/home/hai_ssh/.conda/envs/lunar/bin/python
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
J="work_travel_home_banking"
OUT="${HERE}/results/banking_t3"; LOG="${HERE}/logs/t3"; mkdir -p "${OUT}" "${LOG}"
declare -A DATA=([in]="${HERE}/data/in_banking_100.json" [out]="${HERE}/data/out_banking77_1000.json")
CB="${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA"
RCK="${MC}/outputs/repro_clinc_lunar_bs64/seed_0/t3_banking/lora_moe.pt"
O3DIR="${REPO}/unlearning_o3/unlearn_recent_o3/SCALE_0.1_seed_0_o_unlearn_lora_force_checkpoints_clinc9_origin"
O3LORA="${O3DIR}/lora_force_random_work_force_travel_force_home_force_banking_force"
OOD="${REPO}/ood_ocsvm/ood_checkpoints_clincqa_recent/ood_checkpoints_clincqa_0/"
declare -A T=([testSD]="${D}/clincqa_SD_10/clincqa_${J}_test_SD.json" [testRD]="${D}/clincqa_RD_10/clincqa_not_${J}_test_RD.json" \
              [mrpc]="${D}/mrpcqa/mrpc_test.json" [rte]="${D}/rteqa/rte_validation.json")
log() { echo "[$(date '+%F %T')] $*"; }

o3_eval() {  # $1 out json  $2 split  $3.. extra args
  local out="$1" sp="$2"; shift 2
  ( cd "${REPO}" && PYTHONPATH=/home/hai_ssh/pylibs/peft0181 "${PY}" "${REPO}/quant_attack/eval_quant_o3_clinc.py" \
      --unlearn_topic "banking_${sp}" --test_dataset "${T[$sp]}" --base_model "${CB}" --seed 0 --lora_weights "${O3LORA}" \
      --ood_type banking --ood_setting C --ood_setting_name clincqa --ood_weights "${OOD}" \
      --quant_mode none --result_file "${out}" "$@" ) > "${out%.json}.log" 2> "${out%.json}.err"
}

# ---- 1) O3 smoke ----
SM="${OUT}/o3_smoke_none_testSD.json"
if [[ ! -s "${SM}" ]]; then
  log "1) O3 t3 smoke (testSD 50, 공격 없음)"
  o3_eval "${SM}" testSD --max_samples 50
fi
"${PY}" - "${SM}" "${O3DIR}/test_seed0_banking_testSD.json" <<'EOF' || { log "O3 smoke 불일치 → 중단"; exit 1; }
import json, sys
a = json.load(open(sys.argv[1]))["results"]; b = json.load(open(sys.argv[2]))["results"][:len(a)]
r = sum(x["pred"] == y["pred"] for x, y in zip(a, b)) / len(a)
p = sum(x["prompt"] == y["prompt"] for x, y in zip(a, b)) / len(a)
print(f"[smoke] n={len(a)} prompt 일치 {p:.2f} pred 일치 {r:.2f}")
sys.exit(0 if r >= 0.9 and p == 1.0 else 1)
EOF

# ---- 2) RULE t3 ----
rule() {  # $1 target $2 data $3 epochs
  local tag="rule_$1_$2_ep$3" out="${OUT}/rule_$1_$2_ep$3.json"
  [[ -s "${out}" ]] && { log "[skip] ${tag}"; return; }
  log "2) ${tag}"
  ( cd "${MC}" && "${PY}" "${HERE}/relearn_clinc.py" --config-name forget \
      data_name=clincqa model_family=llama3.1-8b-instruct model_path="${CB}" seed=0 \
      task_id=clincqa_t3_banking layer_modified="[16]" eval_batch_size=16 \
      data_path="${D}/clincqa_SD_only_200/clincqa_banking_train_SD.json" \
      r_data_path="${D}/clincqa_RD_only_200_c/clincqa_not_${J}_train_RD.json" \
      use_unverified=false use_harmful=true \
      sampled_data_path="${T[testSD]}" retain_data_path="${T[testRD]}" \
      utiliy_data_path1="${T[mrpc]}" utiliy_data_path2="${T[rte]}" \
      load_unlearned_model=false save_unlearned_model=false hydra.run.dir="${LOG}/hydra/${tag}" \
      +attack.target="$1" +attack.data="${DATA[$2]}" +attack.epochs="$3" +attack.lr=1e-4 \
      +attack.moe_ckpt="${RCK}" +attack.out_file="${out}" ) > "${LOG}/${tag}.log" 2>&1
  log "   rc=$? $(grep '\[DONE\]' "${LOG}/${tag}.log" | sed 's#.*json | ##')"
}
for t in backbone moe; do for d in in out; do for e in 1 5; do rule "$t" "$d" "$e"; done; done; done

# ---- 3), 4) O3 t3 ----
o3() {  # $1 tag(o3_backbone|o3_lora)  $2 data  $3 epochs  $4 extra flag
  local tag="$1_$2_ep$3" st="${OUT}/$1_$2_ep$3_attack.pt"
  for sp in testSD testRD mrpc rte; do
    local out="${OUT}/${tag}_${sp}.json"; [[ -s "${out}" ]] && continue
    local atk=(--attack_state "${st}")
    [[ ! -s "${st}" ]] && atk=(--attack_train_data "${DATA[$2]}" --attack_epochs "$3" --attack_lr 1e-4 --attack_save "${st}" ${4:+$4})
    log "$( [[ $1 == o3_backbone ]] && echo 3 || echo 4)) ${tag} ${sp} (${atk[0]})"
    o3_eval "${out}" "${sp}" "${atk[@]}"
    log "   rc=$? exact=$("${PY}" -c "import json;print(round(json.load(open('${out}'))['exact_acc'],2))" 2>/dev/null)"
  done
}
for d in in out; do for e in 1 5; do o3 o3_backbone "$d" "$e" ""; done; done
for d in in out; do for e in 1 5; do o3 o3_lora "$d" "$e" "--attack_train_o3_lora"; done; done
log "all done"
