#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
MAIN_ROOT="${MAIN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1/full_seed20260729}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_uncertainty_v1/full_seed20260729}"
SEEDS=(20260730 20260731 20260801)

while [[ ! -f "${RUN_ROOT}/MATRIX_COMPLETE.json" ]]; do sleep 60; done
cd "${PROJECT_ROOT}"

for severity in 0 4 8 16; do
  if [[ "${severity}" == "0" ]]; then
    active_root="${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1"
    active_writeback="writeback/evaluation/writeback.jsonl"
  else
    active_root="${MAIN_ROOT}/severity${severity}"
    active_writeback="writeback/writeback.jsonl"
  fi
  for quantile in 05 15 30; do
    notool_args=()
    uncertainty_args=()
    active_args=()
    for seed in "${SEEDS[@]}"; do
      notool_args+=(
        --baseline
        "${seed}=${active_root}/seed${seed}/notool/${active_writeback}"
      )
      uncertainty_args+=(
        --candidate
        "${seed}=${RUN_ROOT}/severity${severity}/seed${seed}/q${quantile}/writeback/writeback.jsonl"
      )
      active_args+=(
        --candidate
        "${seed}=${active_root}/seed${seed}/benefit/${active_writeback}"
      )
    done
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
        "${RUN_ROOT}/severity${severity}/q${quantile}_vs_notool.json" \
        "${notool_args[@]}" "${uncertainty_args[@]}" \
        --repetitions 5000 --seed 20260729 --split val

    uncertainty_baseline=()
    for seed in "${SEEDS[@]}"; do
      uncertainty_baseline+=(
        --baseline
        "${seed}=${RUN_ROOT}/severity${severity}/seed${seed}/q${quantile}/writeback/writeback.jsonl"
      )
    done
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
        "${RUN_ROOT}/severity${severity}/active_vs_q${quantile}.json" \
        "${uncertainty_baseline[@]}" "${active_args[@]}" \
        --repetitions 5000 --seed 20260729 --split val
  done
done

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" scripts/summarize_sn7_controller_uncertainty_corruption.py \
    "${RUN_ROOT}" "${RUN_ROOT}/summary.json"

printf '{"status":"complete","test_assets_read":false}\n' \
  >"${RUN_ROOT}/SUMMARY_COMPLETE.json"
