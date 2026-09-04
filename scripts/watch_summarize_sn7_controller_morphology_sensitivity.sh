#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
CORE_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_sensitivity_v1"
SEEDS=(20260730 20260731 20260801)
CONDITIONS=(erode2 dilate2 erode8 dilate8)
LOCK_FILE="${RUN_ROOT}/.summary.lock"

mkdir -p "${RUN_ROOT}"
exec 9>"${LOCK_FILE}"
flock -n 9 || exit 0

failure="${RUN_ROOT}/SUMMARY_FAILED.json"
rm -f "${failure}"
record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    printf '{"status":"failed","stage":"sensitivity_summary","exit_code":%d,"test_assets_read":false}\n' \
      "${rc}" >"${failure}.tmp.$$"
    mv -f "${failure}.tmp.$$" "${failure}"
  fi
  exit "${rc}"
}
trap record_failure EXIT

for condition in "${CONDITIONS[@]}"; do
  while [[ ! -s "${RUN_ROOT}/${condition}/MATRIX_COMPLETE.json" ]]; do
    [[ ! -s "${RUN_ROOT}/${condition}/FAILED.json" ]] || exit 5
    sleep 60
  done
done

cd "${PROJECT_ROOT}"
for condition in "${CONDITIONS[@]}"; do
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

assets="${RUN_ROOT}/paper_assets"
if [[ ! -s "${assets}/manifest.json" ]]; then
  [[ ! -e "${assets}" ]] || mv "${assets}" "${assets}.incomplete.$(date +%s)"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/build_sn7_controller_intervention_assets.py \
      "${assets}" \
      --condition "erode2=${RUN_ROOT}/erode2" \
      --condition "erode4=${CORE_ROOT}/erode4" \
      --condition "erode8=${RUN_ROOT}/erode8" \
      --condition "dilate2=${RUN_ROOT}/dilate2" \
      --condition "dilate4=${CORE_ROOT}/dilate4" \
      --condition "dilate8=${RUN_ROOT}/dilate8"
fi
if [[ ! -s "${RUN_ROOT}/PAPER_ASSETS_AUDIT.json" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/audit_sn7_controller_intervention_bundle.py \
      "${assets}" "${RUN_ROOT}/PAPER_ASSETS_AUDIT.json"
fi

printf '{"status":"complete","conditions":6,"policy_seeds":3,"bootstrap_repetitions":5000,"test_assets_read":false}\n' \
  >"${RUN_ROOT}/SUMMARY_COMPLETE.json"
trap - EXIT
