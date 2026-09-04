#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_sensitivity_v1"
mkdir -p "${STORAGE_ROOT}/logs"

launch_condition() {
  local morphology="$1"
  local radius="$2"
  local gpu="$3"
  local seed session log
  for seed in 20260731 20260801; do
    session="sn7_prior_morph_${morphology}${radius}_s${seed}_sidecar"
    log="${STORAGE_ROOT}/logs/${session}.log"
    tmux has-session -t "${session}" 2>/dev/null && continue
    tmux new-session -d -s "${session}" \
      "cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${gpu} RUN_ROOT=${RUN_ROOT} MORPHOLOGY=${morphology} RADIUS=${radius} SEED=${seed} bash scripts/run_sn7_controller_prior_morphology_condition_seed.sh > ${log} 2>&1"
  done
}

launch_condition erode 2 0
launch_condition dilate 2 1
launch_condition erode 8 3
launch_condition dilate 8 5
