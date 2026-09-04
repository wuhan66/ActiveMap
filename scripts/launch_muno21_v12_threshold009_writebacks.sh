#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
SEEDS=(20260822 20260823 20260824 20260825)
GPUS=(1 2 3 4)
THRESHOLD_TAG="${MUNO21_V12_THRESHOLD_TAG:-threshold009}"
SHORT_TAG="${MUNO21_V12_THRESHOLD_SHORT_TAG:-t009}"
LOG_ROOT="${STORAGE_ROOT}/logs/muno21_v12_${THRESHOLD_TAG}_writeback_v1"
mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPUS[$index]}"
  output="${STORAGE_ROOT}/runs/agent/muno21_v12_${THRESHOLD_TAG}_writeback_v1/seed${seed}"
  [[ ! -s "${output}/COMPLETE.json" ]] || continue
  session="muno21_v12_${SHORT_TAG}_wb_s${seed}"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    tmux new-session -d -s "${session}" \
      "cd ${PROJECT_ROOT} && MUNO21_V12_THRESHOLD_TAG=${THRESHOLD_TAG} SEED=${seed} GPU=${gpu} bash scripts/run_muno21_v12_threshold009_writeback_seed.sh > ${LOG_ROOT}/seed${seed}.log 2>&1"
  fi
done
