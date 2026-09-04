#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
MAIN_ROOT="${MAIN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1/full_seed20260729}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_uncertainty_v1/full_seed20260729}"
GPU_IDS=(${GPU_IDS:-1 2 3 4 5})
SEVERITIES=(0 4 8 16)
SEEDS=(20260730 20260731 20260801)
QUANTILES=(05 15 30)

test -f "${MAIN_ROOT}/MATRIX_COMPLETE.json"
mkdir -p "${RUN_ROOT}"

run_one() {
  local severity="$1"
  local seed="$2"
  local quantile="$3"
  local gpu="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    SEVERITY="${severity}" SEED="${seed}" QUANTILE="${quantile}" \
    bash "${PROJECT_ROOT}/scripts/run_sn7_controller_prior_corruption_uncertainty_job.sh"
}

jobs=()
for severity in "${SEVERITIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for quantile in "${QUANTILES[@]}"; do
      jobs+=("${severity}:${seed}:${quantile}")
    done
  done
done

# One clean sentinel verifies the frozen gate and writeback path.
run_one 0 20260730 05 "${GPU_IDS[0]}"

index=1
while (( index < ${#jobs[@]} )); do
  pids=()
  labels=()
  for gpu in "${GPU_IDS[@]}"; do
    (( index < ${#jobs[@]} )) || break
    IFS=: read -r severity seed quantile <<<"${jobs[$index]}"
    run_one "${severity}" "${seed}" "${quantile}" "${gpu}" &
    pids+=("$!")
    labels+=("${severity}:${seed}:q${quantile}:gpu${gpu}")
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

printf '{"status":"complete","jobs":36,"test_assets_read":false}\n' \
  >"${RUN_ROOT}/MATRIX_COMPLETE.json"
