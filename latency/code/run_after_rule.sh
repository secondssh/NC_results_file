#!/usr/bin/env bash
# run_infer.sh (GPU0) + run_train_profile.sh (GPU1) 가 끝나고 GPU 2장이 비면: baseline (2 GPU) → O3 추론 (GPU0)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(dirname "$(dirname "${HERE}")")"
while pgrep -f "latency/run_infer.sh|latency/run_train_profile.sh|measure_infer.py|profile_train.py" > /dev/null; do sleep 30; done
while [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; do echo "[$(date '+%T')] GPU 사용 중 — 대기"; sleep 60; done
echo "[$(date '+%F %T')] RULE 측정 끝, GPU 비었음 → baseline"
bash "${HERE}/run_baseline_train.sh" > "${HERE}/logs/baseline.log" 2>&1
echo "[$(date '+%F %T')] baseline 끝 → O3 추론 (GPU0)"
( cd "${REPO}" && CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/hai_ssh/pylibs/peft0181 /home/hai_ssh/.conda/envs/lunar/bin/python \
    "${HERE}/measure_infer_o3.py" --steps 0 8 ) > "${HERE}/logs/infer_o3.log" 2>&1
echo "[$(date '+%F %T')] O3 rc=$?"; grep -E "^\[o3|Traceback|Error" "${HERE}/logs/infer_o3.log" | tail -12
echo "[$(date '+%F %T')] all finished"
