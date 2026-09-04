#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
GPU="${GPU:-4}"
SEED="${SEED:-20260901}"
DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/direct_vlm_sft_v1"
QUEUE_ROOT="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_sft_v1_seed${SEED}"

mkdir -p "${QUEUE_ROOT}"
exec 8>"${QUEUE_ROOT}/.queue.lock"
flock -n 8 || exit 0
[[ ! -s "${QUEUE_ROOT}/COMPLETE.json" ]] || exit 0

while [[ ! -s "${DATA_ROOT}/train.jsonl" || ! -s "${DATA_ROOT}/val.jsonl" ]]; do
  sleep 30
done
while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

cd "${PROJECT_ROOT}"
exec env GPU="${GPU}" SEED="${SEED}" \
  bash scripts/watch_train_muno21_direct_vlm_sft.sh
