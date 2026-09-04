#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
BASE="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1"
RUN_ROOT="${BASE}/full_seed20260729"
GPU_IDS=(4 5)

# Reuse the two sensitivity GPUs only after their four predeclared jobs finish.
for corruption_seed in 20260730 20260731; do
  for variant in notool benefit; do
    marker="${BASE}/corruption_seed${corruption_seed}/severity16/seed20260730/${variant}/COMPLETE.json"
    while [[ ! -f "${marker}" ]]; do sleep 30; done
  done
done

run_one() {
  local seed="$1"
  local variant="$2"
  local gpu="$3"
  local marker="${RUN_ROOT}/severity16/seed${seed}/${variant}/COMPLETE.json"
  if [[ -f "${marker}" ]]; then
    echo "already complete 16:${seed}:${variant}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" RUN_ROOT="${RUN_ROOT}" \
    CORRUPTION_SEED=20260729 SEVERITY=16 SEED="${seed}" VARIANT="${variant}" \
    bash "${PROJECT_ROOT}/scripts/run_sn7_controller_prior_corruption_job.sh"
}

# Work backward from the tail while the primary watcher advances from the head.
jobs=(
  20260801:benefit
  20260801:forced
  20260801:notool
  20260731:benefit
)
index=0
while (( index < ${#jobs[@]} )); do
  pids=()
  labels=()
  for gpu in "${GPU_IDS[@]}"; do
    (( index < ${#jobs[@]} )) || break
    IFS=: read -r seed variant <<<"${jobs[$index]}"
    run_one "${seed}" "${variant}" "${gpu}" &
    pids+=("$!")
    labels+=("16:${seed}:${variant}:gpu${gpu}")
    ((index += 1))
  done
  status=0
  for job_index in "${!pids[@]}"; do
    if wait "${pids[$job_index]}"; then
      echo "completed ${labels[$job_index]}"
    else
      echo "failed ${labels[$job_index]}" >&2
      status=1
    fi
  done
  (( status == 0 )) || exit "${status}"
done

printf '{"status":"complete","jobs":4,"test_assets_read":false}\n' \
  >"${RUN_ROOT}/TAIL_ACCELERATOR_COMPLETE.json"
