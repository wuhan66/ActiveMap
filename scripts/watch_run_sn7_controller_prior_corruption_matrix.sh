#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1/full_seed20260729}"
GPU_IDS=(${GPU_IDS:-1 2 3 4})
SEVERITIES=(4 8 16)
SEEDS=(20260730 20260731 20260801)
VARIANTS=(notool forced benefit)

for severity in "${SEVERITIES[@]}"; do
  while [[ ! -f "${RUN_ROOT}/severity${severity}/COMPLETE.json" ]]; do
    sleep 30
  done
done

run_one() {
  local severity="$1"
  local seed="$2"
  local variant="$3"
  local gpu="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    SEVERITY="${severity}" SEED="${seed}" VARIANT="${variant}" \
    bash "${PROJECT_ROOT}/scripts/run_sn7_controller_prior_corruption_job.sh"
}

# One end-to-end sentinel must pass before the remaining 26 jobs are released.
run_one 4 20260730 notool "${GPU_IDS[0]}"

jobs=()
for severity in "${SEVERITIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      if [[ "${severity}:${seed}:${variant}" != "4:20260730:notool" ]]; then
        jobs+=("${severity}:${seed}:${variant}")
      fi
    done
  done
done

index=0
while (( index < ${#jobs[@]} )); do
  pids=()
  labels=()
  for gpu in "${GPU_IDS[@]}"; do
    (( index < ${#jobs[@]} )) || break
    IFS=: read -r severity seed variant <<<"${jobs[$index]}"
    run_one "${severity}" "${seed}" "${variant}" "${gpu}" &
    pids+=("$!")
    labels+=("${severity}:${seed}:${variant}:gpu${gpu}")
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

printf '{"status":"complete","jobs":27,"test_assets_read":false}\n' \
  >"${RUN_ROOT}/MATRIX_COMPLETE.json"
