#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
QUEUE_ROOT="${STORAGE_ROOT}/logs/muno21_grpo_epoch_scaling_v1"
mkdir -p "${QUEUE_ROOT}"

wait_for_gpu() {
  local gpu="$1"
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    sleep 60
  done
}

run_after_free() {
  local gpu="$1" epochs="$2"
  wait_for_gpu "${gpu}"
  PROJECT_ROOT="${PROJECT_ROOT}" STORAGE_ROOT="${STORAGE_ROOT}" \
    GPU="${gpu}" EPOCHS="${epochs}" SEED=20261003 SERVER_LABEL=hdpi \
    bash "${PROJECT_ROOT}/scripts/run_muno21_grpo_epoch_scaling_job.sh"
}

(run_after_free 1 2) >"${QUEUE_ROOT}/queue_epochs2_seed20261003.log" 2>&1 &
p1=$!
(run_after_free 2 3) >"${QUEUE_ROOT}/queue_epochs3_seed20261003.log" 2>&1 &
p2=$!
wait "${p1}" "${p2}"
date -Is >"${QUEUE_ROOT}/SEED3_QUEUE_COMPLETED"
