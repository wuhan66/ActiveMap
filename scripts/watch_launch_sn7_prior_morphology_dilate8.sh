#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
CORE_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_sensitivity_v1"
GPU="${DILATE8_GPU:-2}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOCK_FILE="${RUN_ROOT}/.dilate8_launcher.lock"

mkdir -p "${RUN_ROOT}" "${STORAGE_ROOT}/logs"
exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

while [[ ! -s "${CORE_ROOT}/dilate4/COMPLETE.json" ||
         ! -s "${CORE_ROOT}/dilate4/PARITY_AUDIT.json" ]]; do
  [[ ! -s "${CORE_ROOT}/dilate4/FAILED.json" ]] || exit 5
  sleep "${POLL_SECONDS}"
done

session="sn7_prior_morph_dilate8_sensitivity"
log="${STORAGE_ROOT}/logs/${session}.log"
if [[ ! -s "${RUN_ROOT}/dilate8/MATRIX_COMPLETE.json" ]] &&
   ! tmux has-session -t "${session}" 2>/dev/null; then
  tmux new-session -d -s "${session}" \
    "cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${GPU} RUN_ROOT=${RUN_ROOT} MORPHOLOGY=dilate RADIUS=8 bash scripts/run_sn7_controller_prior_morphology_condition.sh > ${log} 2>&1"
fi
