#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
WAIT_FOR="${STORAGE_ROOT}/runs/agent/muno21_grpo_epoch_scaling_v1/epochs2_seed20261002/COMPLETED.json"
TARGET="${STORAGE_ROOT}/runs/agent/muno21_grpo_epoch_scaling_v1/epochs5_seed20261003"

while [[ ! -s "${WAIT_FOR}" ]]; do
  sleep 60
done
while [[ -n "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 60
done
[[ ! -e "${TARGET}" ]] || exit 0

PROJECT_ROOT="${PROJECT_ROOT}" STORAGE_ROOT="${STORAGE_ROOT}" \
  GPU=0 EPOCHS=5 SEED=20261003 SERVER_LABEL=hdpi \
  bash "${PROJECT_ROOT}/scripts/run_muno21_grpo_epoch_scaling_job.sh"
