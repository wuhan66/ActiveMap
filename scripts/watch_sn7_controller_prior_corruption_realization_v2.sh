#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_BASE="${RUN_BASE:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_realization_v2}"
CORRUPTION_SEED="${CORRUPTION_SEED:?set CORRUPTION_SEED}"
SEVERITY="${SEVERITY:?set SEVERITY}"
POLL_SECONDS="${POLL_SECONDS:-30}"

case "${CORRUPTION_SEED}" in 20260730|20260731|20260802) ;; *) exit 2 ;; esac
case "${SEVERITY}" in 1|2|4|8|16) ;; *) exit 2 ;; esac

RUN_ROOT="${RUN_BASE}/seed${CORRUPTION_SEED}"
STATE_MARKER="${RUN_ROOT}/severity${SEVERITY}/COMPLETE.json"
COMPLETE_MARKER="${RUN_ROOT}/severity${SEVERITY}/REALIZATION_MATRIX_COMPLETE.json"
STATE_FAILURE="${RUN_ROOT}/severity${SEVERITY}/FAILED.json"
FAILURE_MARKER="${RUN_ROOT}/severity${SEVERITY}/MATRIX_FAILED.json"
rm -f "${FAILURE_MARKER}"
record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    local temporary="${FAILURE_MARKER}.tmp.$$"
    printf \
      '{"status":"failed","stage":"realization_matrix","exit_code":%d,"test_assets_read":false}\n' \
      "${rc}" >"${temporary}"
    mv -f "${temporary}" "${FAILURE_MARKER}"
  fi
  exit "${rc}"
}
trap record_failure EXIT

cd "${PROJECT_ROOT}"
while [[ ! -s "${STATE_MARKER}" ]]; do
  if [[ -s "${STATE_FAILURE}" ]]; then
    cat "${STATE_FAILURE}" >&2
    exit 5
  fi
  sleep "${POLL_SECONDS}"
done

for seed in 20260730 20260731 20260801; do
  for variant in notool forced benefit; do
    job_root="${RUN_ROOT}/severity${SEVERITY}/seed${seed}/${variant}"
    marker="${job_root}/COMPLETE.json"
    audit="${job_root}/AUDIT.json"
    [[ -s "${marker}" && -s "${audit}" ]] && continue
    while [[ -d "${job_root}/writeback" && ! -s "${job_root}/writeback/summary.json" ]]; do
      [[ ! -s "${job_root}/FAILED.json" ]] || {
        cat "${job_root}/FAILED.json" >&2
        exit 6
      }
      sleep "${POLL_SECONDS}"
    done
    RUN_ROOT="${RUN_ROOT}" \
      CORRUPTION_SEED="${CORRUPTION_SEED}" \
      SEVERITY="${SEVERITY}" \
      SEED="${seed}" \
      VARIANT="${variant}" \
      bash scripts/run_sn7_controller_prior_corruption_job.sh
  done
done

temporary="${COMPLETE_MARKER}.tmp.$$"
printf \
  '{"status":"complete","severity":%s,"corruption_seed":%s,"policy_seeds":3,"variants":3,"test_assets_read":false}\n' \
  "${SEVERITY}" "${CORRUPTION_SEED}" >"${temporary}"
mv -f "${temporary}" "${COMPLETE_MARKER}"
trap - EXIT
