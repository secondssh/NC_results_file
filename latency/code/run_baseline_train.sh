#!/usr/bin/env bash
# baseline full-FT (GA / GD(=ga_gd) / NPO+KL) 학습 시간 + GPU 메모리: 원래 설정 그대로 (2 GPU DeepSpeed ZeRO-3, optimizer CPU offload,
# global batch 128, lr 1e-5, CLINC max_len 256 / TOFU 512), t0 만, 1 epoch (원래 10 epoch → epoch 당 시간 × 10 으로 비교).
# GPU 메모리 = nvidia-smi 1초 간격 polling 최대값 (GPU 별), 시간 = HF train_runtime. 저장된 모델(16GB)은 끝나면 지움.
# 사용 (modify_code 에서): setsid nohup bash latency/run_baseline_train.sh > latency/logs/baseline.log 2>&1 < /dev/null & disown
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MC="$(dirname "${HERE}")"; REPO="$(dirname "${MC}")"
BL="${REPO}/unlearning_baseline_new"; D="${REPO}/data_new"
PY=/home/hai_ssh/.conda/envs/lunar/bin/python; export PATH="/home/hai_ssh/.conda/envs/lunar/bin:${PATH}"
SCR="/tmp/claude-10011/-home-hai-ssh/1484149c-2ae3-43d6-be42-03577263b907/scratchpad/baseline_ckpt"
OUT="${HERE}/results/baseline"; mkdir -p "${OUT}" "${SCR}"
export CUDA_VISIBLE_DEVICES=0,1
for job in "clinc ga" "clinc ga_gd" "clinc npo_kl" "tofu ga" "tofu ga_gd" "tofu npo_kl"; do
  read -r ds m <<< "${job}"
  tag="${ds}_${m}_t0"; [[ -s "${OUT}/${tag}.json" ]] && { echo "[skip] ${tag}"; continue; }
  if [[ "${ds}" == clinc ]]; then
    base="${MC}/train31Llama/trained_model/O3_LLAMA31_ClincQA"; fg="${D}/clincqa_SD_only_200/clincqa_work_train_SD.json"
    rt="${D}/clincqa_RD_only_200_c/clincqa_not_work_train_RD.json"; ml=256
  else
    base="${MC}/train31Llama/trained_model/O3_LLAMA31_TOFU"; fg="${D}/ptofu_SD_only/tofu_AG01_SD.json"
    rt="${D}/ptofu_RD/tofu_step001_RD.json"; ml=512
  fi
  args=(--method "${m}" --base_model "${base}" --forget_dataset "${fg}" --output_dir "${SCR}/${tag}" --resume_from_checkpoint "${base}"
        --seed 0 --epochs 1 --per_device_batch_size 1 --grad_accum 64 --max_length "${ml}" --lr 1e-5 --beta 0.1
        --backend deepspeed --deepspeed "${BL}/ds_zero3.json")
  [[ "${m}" != ga ]] && args+=(--retain_dataset "${rt}")
  echo "[$(date '+%F %T')] ${tag}"
  ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits; sleep 1; done ) > "${OUT}/${tag}.smi" &
  POLL=$!
  start=$(date +%s)
  ( cd "${BL}" && "${PY}" -m torch.distributed.run --nproc_per_node=2 --master_port 29561 "${BL}/train_fullft.py" "${args[@]}" ) > "${HERE}/logs/baseline_${tag}.log" 2>&1
  rc=$?; wall=$(( $(date +%s) - start )); kill ${POLL} 2>/dev/null
  rt_s=$(grep -ao "'train_runtime': [0-9.]*" "${HERE}/logs/baseline_${tag}.log" | tail -1 | awk '{print $2}')
  steps=$(grep -ao "'epoch': [0-9.]*" "${HERE}/logs/baseline_${tag}.log" | wc -l)
  "${PY}" - "${OUT}/${tag}.smi" "${OUT}/${tag}.json" "${tag}" "${rc}" "${wall}" "${rt_s:-nan}" "${steps}" <<'P'
import sys, json, collections
smi, out, tag, rc, wall, rt, steps = sys.argv[1:]
peak = collections.defaultdict(int)
for l in open(smi):
    try:
        i, m = [int(x) for x in l.split(",")]; peak[i] = max(peak[i], m)
    except ValueError:
        pass
json.dump({"tag": tag, "rc": int(rc), "wall_sec": int(wall), "train_runtime_sec_1epoch": float(rt),
           "n_logged_steps": int(steps), "nvidia_smi_peak_MiB": dict(peak), "setting": "2 GPU DeepSpeed ZeRO-3 + CPU optimizer offload, global batch 128, 1 epoch"},
          open(out, "w"), indent=1)
print(f"  rc={rc} wall={wall}s train_runtime={rt}s peak={dict(peak)}")
P
  rm -rf "${SCR:?}/${tag}"
done
echo "[$(date '+%F %T')] baseline finished"
