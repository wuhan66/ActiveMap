#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
PRIMARY="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1/full_seed20260729"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_realization_v2"
POLICY_SEEDS=(20260730 20260731 20260801)
CORRUPTION_SEEDS=(20260729 20260730 20260731)
mkdir -p "${RUN_ROOT}"
LOCK_FILE="${RUN_ROOT}/.summary.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "SN7 controller realization summary is already active."
  exit 0
fi
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

for corruption_seed in 20260730 20260731; do
  for severity in 4 8; do
    marker="${RUN_ROOT}/seed${corruption_seed}/severity${severity}/REALIZATION_MATRIX_COMPLETE.json"
    failure="${RUN_ROOT}/seed${corruption_seed}/severity${severity}/MATRIX_FAILED.json"
    while [[ ! -s "${marker}" ]]; do
      [[ ! -s "${failure}" ]] || { cat "${failure}" >&2; exit 5; }
      sleep 60
    done
  done
done

cd "${PROJECT_ROOT}"
for severity in 4 8; do
  output="${RUN_ROOT}/severity${severity}_aggregate"
  mkdir -p "${output}"
  controller_args=()
  notool_args=()
  benefit_args=()
  forced_args=()
  index=1
  for corruption_seed in "${CORRUPTION_SEEDS[@]}"; do
    if [[ "${corruption_seed}" == 20260729 ]]; then
      root="${PRIMARY}/severity${severity}"
    else
      root="${RUN_ROOT}/seed${corruption_seed}/severity${severity}"
    fi
    for policy_seed in "${POLICY_SEEDS[@]}"; do
      label="$((severity * 100 + index))"
      controller_args+=(--notool "${label}=${root}/seed${policy_seed}/notool/closed_loop/edit_utility.jsonl")
      controller_args+=(--forced "${label}=${root}/seed${policy_seed}/forced/closed_loop/edit_utility.jsonl")
      controller_args+=(--benefit "${label}=${root}/seed${policy_seed}/benefit/closed_loop/edit_utility.jsonl")
      notool_args+=(--baseline "${label}=${root}/seed${policy_seed}/notool/writeback/writeback.jsonl")
      forced_args+=(--baseline "${label}=${root}/seed${policy_seed}/forced/writeback/writeback.jsonl")
      benefit_args+=(--candidate "${label}=${root}/seed${policy_seed}/benefit/writeback/writeback.jsonl")
      ((index += 1))
    done
  done
  if [[ ! -s "${output}/controller_summary.json" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/summarize_sn7_step0_three_policy.py \
        "${controller_args[@]}" --split val --bootstrap-repetitions 5000 \
        "${output}/controller_summary.json"
  fi
  if [[ ! -s "${output}/benefit_vs_notool.json" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
        "${output}/benefit_vs_notool.json" "${notool_args[@]}" "${benefit_args[@]}" \
        --repetitions 5000 --seed 20260730 --split val
  fi
  if [[ ! -s "${output}/benefit_vs_forced.json" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
        "${output}/benefit_vs_forced.json" "${forced_args[@]}" "${benefit_args[@]}" \
        --repetitions 5000 --seed 20260730 --split val
  fi
done

if [[ ! -s "${RUN_ROOT}/paper_assets/manifest.json" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/build_sn7_controller_intervention_assets.py \
      "${RUN_ROOT}/paper_assets" \
      --condition "4px=${RUN_ROOT}/severity4_aggregate" \
      --condition "8px=${RUN_ROOT}/severity8_aggregate"
fi
if [[ ! -s "${RUN_ROOT}/PAPER_ASSETS_AUDIT.json" ]]; then
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/audit_sn7_controller_intervention_bundle.py \
      "${RUN_ROOT}/paper_assets" "${RUN_ROOT}/PAPER_ASSETS_AUDIT.json"
fi
printf '{"status":"complete","test_assets_read":false}\n' \
  >"${RUN_ROOT}/SUMMARY_COMPLETE.json"
trap - EXIT
