#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
GPU="${GPU:-4}"
SEED="${SEED:-20260903}"
UPSTREAM="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_sft_v1_seed20260901"
MODEL="${MODEL:-/home/wh/hf_models/gemma-3-4b-it}"
RUN_FAMILY="${RUN_FAMILY:-muno21_direct_vlm_gemma3_4b_sft_v1}"
RUN_ROOT="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${SEED}"

mkdir -p "${RUN_ROOT}"
exec 6>"${RUN_ROOT}/.queue.lock"
flock -n 6 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0

while [[ ! -s "${UPSTREAM}/COMPLETE.json" ]]; do
  [[ ! -s "${UPSTREAM}/FAILED.json" ]] || exit 3
  sleep 60
done
while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

cd "${PROJECT_ROOT}"
exec env GPU="${GPU}" SEED="${SEED}" MODEL="${MODEL}" \
  RUN_FAMILY="${RUN_FAMILY}" \
  bash scripts/watch_train_muno21_direct_vlm_sft_generic.sh
