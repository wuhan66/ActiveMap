#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_BASE="${RUN_BASE:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_dense_v3b_20260802}"
CORRUPTION_SEED="${CORRUPTION_SEED:-20260802}"

launch_job() {
  local severity="$1"
  local gpu="$2"
  local policy_seed="$3"
  local variant="$4"
  local job_root="${RUN_BASE}/seed${CORRUPTION_SEED}/severity${severity}/seed${policy_seed}/${variant}"
  local label="seed${CORRUPTION_SEED}_severity${severity}_policy${policy_seed}_${variant}_parallel"

  [[ ! -s "${job_root}/COMPLETE.json" ]] || {
    echo "Already complete: ${job_root}"
    return 0
  }
  mkdir -p "${RUN_BASE}/logs" "${RUN_BASE}/pids" "${RUN_BASE}/status"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export PROJECT_ROOT STORAGE_ROOT
    export RUN_ROOT="${RUN_BASE}/seed${CORRUPTION_SEED}"
    export CORRUPTION_SEED SEVERITY="${severity}" SEED="${policy_seed}" VARIANT="${variant}"
    printf 'running\n' >"${RUN_BASE}/status/${label}.txt"
    set +e
    bash "${PROJECT_ROOT}/scripts/run_sn7_controller_prior_corruption_job.sh"
    code="$?"
    set -e
    printf '%s\n' "${code}" >"${RUN_BASE}/status/${label}.txt"
    exit "${code}"
  ) >"${RUN_BASE}/logs/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${RUN_BASE}/pids/${label}.pid"
  echo "${label}: GPU ${gpu}, PID $!"
}

# These queue-tail jobs are disjoint from the currently active seed-20260730
# workers. The per-job flock in the worker prevents later watcher duplication.
launch_job 1 2 20260801 benefit
launch_job 2 5 20260801 benefit
