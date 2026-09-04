#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1"
SEEDS=(20260730 20260731 20260801)
mkdir -p "${RUN_ROOT}"
FAILURE_MARKER="${RUN_ROOT}/SUMMARY_FAILED.json"
rm -f "${FAILURE_MARKER}"
record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    local temporary="${FAILURE_MARKER}.tmp.$$"
    printf \
      '{"status":"failed","stage":"summary","exit_code":%d,"test_assets_read":false}\n' \
      "${rc}" >"${temporary}"
    mv -f "${temporary}" "${FAILURE_MARKER}"
  fi
  exit "${rc}"
}
trap record_failure EXIT

for condition in erode4 dilate4; do
  while [[ ! -s "${RUN_ROOT}/${condition}/MATRIX_COMPLETE.json" ]]; do
    for failure in \
      "${RUN_ROOT}/${condition}/FAILED.json" \
      "${RUN_ROOT}/MATRIX_FAILED.json"; do
      [[ ! -s "${failure}" ]] || { cat "${failure}" >&2; exit 5; }
    done
    sleep 60
  done
done

cd "${PROJECT_ROOT}"
for condition in erode4 dilate4; do
  root="${RUN_ROOT}/${condition}"
  controller_args=()
  notool_args=()
  benefit_args=()
  forced_args=()
  for seed in "${SEEDS[@]}"; do
    controller_args+=(--notool "${seed}=${root}/seed${seed}/notool/closed_loop/edit_utility.jsonl")
    controller_args+=(--forced "${seed}=${root}/seed${seed}/forced/closed_loop/edit_utility.jsonl")
    controller_args+=(--benefit "${seed}=${root}/seed${seed}/benefit/closed_loop/edit_utility.jsonl")
    notool_args+=(--baseline "${seed}=${root}/seed${seed}/notool/writeback/writeback.jsonl")
    forced_args+=(--baseline "${seed}=${root}/seed${seed}/forced/writeback/writeback.jsonl")
    benefit_args+=(--candidate "${seed}=${root}/seed${seed}/benefit/writeback/writeback.jsonl")
  done
  if [[ ! -s "${root}/controller_summary.json" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/summarize_sn7_step0_three_policy.py \
        "${controller_args[@]}" --split val --bootstrap-repetitions 5000 \
        "${root}/controller_summary.json"
  fi
  if [[ ! -s "${root}/benefit_vs_notool.json" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
        "${root}/benefit_vs_notool.json" "${notool_args[@]}" "${benefit_args[@]}" \
        --repetitions 5000 --seed 20260730 --split val
  fi
  if [[ ! -s "${root}/benefit_vs_forced.json" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
        "${root}/benefit_vs_forced.json" "${forced_args[@]}" "${benefit_args[@]}" \
        --repetitions 5000 --seed 20260730 --split val
  fi
done

if [[ ! -s "${RUN_ROOT}/paper_assets/manifest.json" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/build_sn7_controller_intervention_assets.py \
      "${RUN_ROOT}/paper_assets" \
      --condition "erode4=${RUN_ROOT}/erode4" \
      --condition "dilate4=${RUN_ROOT}/dilate4"
fi
if [[ ! -s "${RUN_ROOT}/PAPER_ASSETS_AUDIT.json" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/audit_sn7_controller_intervention_bundle.py \
      "${RUN_ROOT}/paper_assets" "${RUN_ROOT}/PAPER_ASSETS_AUDIT.json"
fi
printf '{"status":"complete","test_assets_read":false}\n' \
  >"${RUN_ROOT}/SUMMARY_COMPLETE.json"
trap - EXIT
