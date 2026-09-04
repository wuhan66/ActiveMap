#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_sensitivity_v1}"

mkdir -p "${RUN_ROOT}" "${STORAGE_ROOT}/logs"
cd "${PROJECT_ROOT}"

launch() {
  local morphology="$1"
  local radius="$2"
  local gpu="$3"
  local session="sn7_prior_morph_${morphology}${radius}_sensitivity"
  local log="${STORAGE_ROOT}/logs/${session}.log"
  [[ ! -s "${RUN_ROOT}/${morphology}${radius}/MATRIX_COMPLETE.json" ]] || return 0
  tmux has-session -t "${session}" 2>/dev/null && return 0
  tmux new-session -d -s "${session}" \
    "cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${gpu} RUN_ROOT=${RUN_ROOT} MORPHOLOGY=${morphology} RADIUS=${radius} bash scripts/run_sn7_controller_prior_morphology_condition.sh > ${log} 2>&1"
}

# Keep the simultaneous project allocation at five physical cards while the
# existing erode4/dilate4 jobs occupy GPUs 5 and 2.
launch erode 2 0
launch dilate 2 1
launch erode 8 3
