#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1}"
GPU="${MORPHOLOGY_GPU:-5}"
MAX_LOAD="${MORPHOLOGY_MAX_LOAD:-140}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOCK_FILE="${RUN_ROOT}/.parallel_lanes.lock"

mkdir -p "${RUN_ROOT}" "${STORAGE_ROOT}/logs"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Morphology parallel-lane watcher is already active."
  exit 0
fi

while true; do
  [[ ! -s "${RUN_ROOT}/MATRIX_FAILED.json" ]] || {
    cat "${RUN_ROOT}/MATRIX_FAILED.json" >&2
    exit 5
  }
  load="$(cut -d' ' -f1 /proc/loadavg)"
  if awk -v load="${load}" -v maximum="${MAX_LOAD}" \
    'BEGIN { exit !(load < maximum) }'; then
    break
  fi
  echo "Host load ${load} >= ${MAX_LOAD}; waiting ${POLL_SECONDS}s."
  sleep "${POLL_SECONDS}"
done

cd "${PROJECT_ROOT}"
for seed in 20260731 20260801; do
  session="sn7_prior_morph_lane_s${seed}"
  log="${STORAGE_ROOT}/logs/${session}.log"
  if ! tmux has-session -t "${session}" 2>/dev/null; then
    tmux new-session -d -s "${session}" \
      "cd ${PROJECT_ROOT} && CUDA_VISIBLE_DEVICES=${GPU} POLICY_SEED=${seed} bash scripts/run_sn7_controller_prior_morphology_lane.sh > ${log} 2>&1"
  fi
done
