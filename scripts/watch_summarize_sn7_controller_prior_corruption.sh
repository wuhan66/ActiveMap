#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1/full_seed20260729}"
SEEDS=(20260730 20260731 20260801)

while [[ ! -f "${RUN_ROOT}/MATRIX_COMPLETE.json" ]]; do sleep 60; done
cd "${PROJECT_ROOT}"

for severity in 4 8 16; do
  root="${RUN_ROOT}/severity${severity}"
  controller_args=()
  notool_args=()
  benefit_args=()
  forced_args=()
  for seed in "${SEEDS[@]}"; do
    controller_args+=(--notool "${seed}=${root}/seed${seed}/notool/closed_loop/edit_utility.jsonl")
    controller_args+=(--forced "${seed}=${root}/seed${seed}/forced/closed_loop/edit_utility.jsonl")
    controller_args+=(--benefit "${seed}=${root}/seed${seed}/benefit/closed_loop/edit_utility.jsonl")
    notool_args+=(--baseline "${seed}=${root}/seed${seed}/notool/writeback/writeback.jsonl")
    benefit_args+=(--candidate "${seed}=${root}/seed${seed}/benefit/writeback/writeback.jsonl")
    forced_args+=(--baseline "${seed}=${root}/seed${seed}/forced/writeback/writeback.jsonl")
  done
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/summarize_sn7_step0_three_policy.py \
      "${controller_args[@]}" --split val --bootstrap-repetitions 5000 \
      "${root}/controller_summary.json"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
      "${root}/benefit_vs_notool.json" "${notool_args[@]}" "${benefit_args[@]}" \
      --repetitions 5000 --seed 20260729 --split val
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
      "${root}/benefit_vs_forced.json" "${forced_args[@]}" "${benefit_args[@]}" \
      --repetitions 5000 --seed 20260729 --split val
done

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" scripts/summarize_sn7_controller_prior_corruption.py \
    "${RUN_ROOT}" \
    "${STORAGE_ROOT}/runs/sn7_active_catalog/sn7_step0_selective_tool_frontier_three_seed_v2.json" \
    "${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1/benefit_vs_notool.json" \
    "${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1/benefit_vs_forced.json" \
    "${RUN_ROOT}/summary.json"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" scripts/plot_sn7_controller_prior_corruption.py \
    "${RUN_ROOT}/summary.json" "${RUN_ROOT}/figures"

printf '{"status":"complete","test_assets_read":false}\n' \
  >"${RUN_ROOT}/SUMMARY_COMPLETE.json"
