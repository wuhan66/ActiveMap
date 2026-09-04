#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_uncertainty_v1/full_seed20260729}"
PARENT_PID="${PARENT_PID:?set paused scheduler PID}"
GPU_SLOTS=(1 2 3 4 5 1 2 3 4 5)
SEVERITIES=(0 4 8 16)
SEEDS=(20260730 20260731 20260801)
QUANTILES=(05 15 30)

cd "${PROJECT_ROOT}"

# Existing children continue while only the parent scheduler is paused.
while pgrep -f \
  "^bash .*/run_sn7_controller_prior_corruption_uncertainty_job.sh$" \
  >/dev/null; do
  sleep 20
done

jobs=()
for severity in "${SEVERITIES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for quantile in "${QUANTILES[@]}"; do
      complete="${RUN_ROOT}/severity${severity}/seed${seed}/q${quantile}/COMPLETE.json"
      if [[ ! -f "${complete}" ]]; then
        jobs+=("${severity}:${seed}:${quantile}")
      fi
    done
  done
done

if (( ${#jobs[@]} > ${#GPU_SLOTS[@]} )); then
  echo "tail has ${#jobs[@]} jobs but only ${#GPU_SLOTS[@]} slots" >&2
  exit 2
fi

pids=()
labels=()
for index in "${!jobs[@]}"; do
  IFS=: read -r severity seed quantile <<<"${jobs[$index]}"
  gpu="${GPU_SLOTS[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu}" \
    SEVERITY="${severity}" SEED="${seed}" QUANTILE="${quantile}" \
    bash "${PROJECT_ROOT}/scripts/run_sn7_controller_prior_corruption_uncertainty_job.sh" &
  pids+=("$!")
  labels+=("${severity}:${seed}:q${quantile}:gpu${gpu}")
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "accelerated-complete ${labels[$index]}"
  else
    echo "accelerated-failed ${labels[$index]}" >&2
    status=1
  fi
done

if (( status == 0 )); then
  kill -CONT "${PARENT_PID}"
  printf '{"status":"complete","jobs":%s,"scheduler_resumed":true,"test_assets_read":false}\n' \
    "${#jobs[@]}" >"${RUN_ROOT}/TAIL_ACCELERATOR_COMPLETE.json"
else
  echo "scheduler remains paused after accelerator failure" >&2
  exit "${status}"
fi
