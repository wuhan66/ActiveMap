#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
LOG_ROOT="${STORAGE_ROOT}/logs/muno21_grpo_epoch_scaling_v1"
mkdir -p "${LOG_ROOT}"

run_one() {
  local gpu="$1" epochs="$2" seed="$3"
  PROJECT_ROOT="${PROJECT_ROOT}" STORAGE_ROOT="${STORAGE_ROOT}" \
    GPU="${gpu}" EPOCHS="${epochs}" SEED="${seed}" SERVER_LABEL=hdpi \
    bash "${PROJECT_ROOT}/scripts/run_muno21_grpo_epoch_scaling_job.sh"
}

(run_one 0 2 20261002) >"${LOG_ROOT}/queue_epochs2_seed20261002.log" 2>&1 & p0=$!
(run_one 1 3 20261002) >"${LOG_ROOT}/queue_epochs3_seed20261002.log" 2>&1 & p1=$!
(run_one 2 4 20261001) >"${LOG_ROOT}/queue_epochs4_seed20261001.log" 2>&1 & p2=$!
(run_one 3 4 20261002) >"${LOG_ROOT}/queue_epochs4_seed20261002.log" 2>&1 & p3=$!

wait "${p0}" "${p1}" "${p2}" "${p3}"
date -Is >"${LOG_ROOT}/MISSING_FOUR_COMPLETED"
