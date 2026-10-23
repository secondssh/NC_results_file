#!/usr/bin/env bash
# baseline full-FT (GA / GD(=ga_gd) / NPO+KL) 학습 시간 + 메모리 v2: run_baseline_train.sh 와 같은 설정 (원래 baseline 설정 그대로:
# 2 GPU DeepSpeed ZeRO-3, optimizer CPU offload, global batch 128, lr 1e-5, CLINC max_len 256 / TOFU 512), t0, 1 epoch.
# v2: monitor.py 로 감싸 GPU (nvidia-smi peak, GPU 별) + host RAM (프로세스 트리 RSS peak) + load average 기록.
# 시간 = HF train_runtime (학습 루프만, 모델 로드 / DeepSpeed 초기화 제외) + wall. 저장된 모델(16GB)은 끝나면 지움.
# 사용 (modify_code 에서): JOBS="clinc ga_gd" bash latency/run_baseline_train_v2.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MC="$(dirname "${HERE}")"; REPO="$(dirname "${MC}")"
BL="${REPO}/unlearning_baseline_new"; D="${REPO}/data_new"
PY=/home/hai_ssh/.conda/envs/lunar/bin/python; export PATH="/home/hai_ssh/.conda/envs/lunar/bin:${PATH}"
SCR="/tmp/claude-10011/-home-hai-ssh/1484149c-2ae3-43d6-be42-03577263b907/scratchpad/baseline_ckpt"
OUT="${OUT:-${HERE}/results/v2/baseline}"; LOG="${HERE}/logs/v2"; mkdir -p "${OUT}" "${SCR}" "${LOG}"
export CUDA_VISIBLE_DEVICES=0,1
JOBS="${JOBS:-clinc ga_gd;clinc ga;clinc npo_kl;tofu ga_gd;tofu ga;tofu npo_kl}"
IFS=';' read -r -a JOB_LIST <<< "${JOBS}"
for job in "${JOB_LIST[@]}"; do
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
  start=$(date +%s)
  ( cd "${BL}" && "${PY}" "${HERE}/monitor.py" --out "${OUT}/${tag}_monitor.json" -- \
      "${PY}" -m torch.distributed.run --nproc_per_node=2 --master_port 29561 "${BL}/train_fullft.py" "${args[@]}" ) \
      > "${LOG}/baseline_${tag}.log" 2>&1
  rc=$?; wall=$(( $(date +%s) - start ))
  rt_s=$(grep -ao "'train_runtime': [0-9.]*" "${LOG}/baseline_${tag}.log" | tail -1 | awk '{print $2}')
  "${PY}" - "${OUT}/${tag}_monitor.json" "${OUT}/${tag}.json" "${tag}" "${rc}" "${wall}" "${rt_s:-nan}" <<'P'
import sys, json
mon, out, tag, rc, wall, rt = sys.argv[1:]
m = json.load(open(mon))
json.dump({"tag": tag, "rc": int(rc), "wall_sec": int(wall), "train_runtime_sec_1epoch": float(rt),
           "nvidia_smi_peak_MiB": m["gpu_peak_MiB"], "host_pss_peak_MB": m["host_pss_peak_MB"], "host_rss_peak_MB": m["host_rss_peak_MB"],
           "clean": m["clean"], "foreign_gpu_pids": m["foreign_gpu_pids"],
           "load1_max": m["load1_max"], "load1_mean": m["load1_mean"], "n_gpu": 2,
           "setting": "2 GPU DeepSpeed ZeRO-3 + CPU optimizer offload, global batch 128, 1 epoch"}, open(out, "w"), indent=1)
print(f"  rc={rc} wall={wall}s train_runtime={rt}s gpu_peak={m['gpu_peak_MiB']} host_pss_peak={m['host_pss_peak_MB']:.0f}MB clean={m['clean']}")
P
  rm -rf "${SCR:?}/${tag}"
done
echo "[$(date '+%F %T')] baseline v2 finished"
