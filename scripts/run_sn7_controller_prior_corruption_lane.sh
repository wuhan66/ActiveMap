#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_BASE="${RUN_BASE:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_realization_v2}"
CORRUPTION_SEED="${CORRUPTION_SEED:?set CORRUPTION_SEED}"
SEVERITY="${SEVERITY:?set SEVERITY}"
POLICY_SEED="${POLICY_SEED:?set POLICY_SEED}"

case "${CORRUPTION_SEED}" in 20260730|20260731) ;; *) exit 2 ;; esac
case "${SEVERITY}" in 4|8) ;; *) exit 2 ;; esac
case "${POLICY_SEED}" in 20260731|20260801) ;; *) exit 2 ;; esac

RUN_ROOT="${RUN_BASE}/seed${CORRUPTION_SEED}"
FAILURE_MARKER="${RUN_ROOT}/severity${SEVERITY}/lane_seed${POLICY_SEED}_FAILED.json"
rm -f "${FAILURE_MARKER}"
record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    local temporary="${FAILURE_MARKER}.tmp.$$"
    printf \
      '{"status":"failed","stage":"parallel_policy_lane","corruption_seed":%s,"severity":%s,"policy_seed":%s,"exit_code":%d,"test_assets_read":false}\n' \
      "${CORRUPTION_SEED}" "${SEVERITY}" "${POLICY_SEED}" "${rc}" >"${temporary}"
    mv -f "${temporary}" "${FAILURE_MARKER}"
  fi
  exit "${rc}"
}
trap record_failure EXIT

cd "${PROJECT_ROOT}"
for variant in notool forced benefit; do
  RUN_ROOT="${RUN_ROOT}" CORRUPTION_SEED="${CORRUPTION_SEED}" \
    SEVERITY="${SEVERITY}" SEED="${POLICY_SEED}" VARIANT="${variant}" \
    bash scripts/run_sn7_controller_prior_corruption_job.sh
done
trap - EXIT
