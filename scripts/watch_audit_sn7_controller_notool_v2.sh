#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_realization_v2"
FAILURE_MARKER="${RUN_ROOT}/NO_TOOL_AUDIT_FAILED.json"
rm -f "${FAILURE_MARKER}"

record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    local temporary="${FAILURE_MARKER}.tmp.$$"
    printf \
      '{"status":"failed","stage":"notool_retroactive_audit","exit_code":%d,"test_assets_read":false}\n' \
      "${rc}" >"${temporary}"
    mv -f "${temporary}" "${FAILURE_MARKER}"
  fi
  exit "${rc}"
}
trap record_failure EXIT

cd "${PROJECT_ROOT}"
for corruption_seed in 20260730 20260731; do
  for severity in 4 8; do
    for policy_seed in 20260730 20260731 20260801; do
      job="${RUN_ROOT}/seed${corruption_seed}/severity${severity}/seed${policy_seed}/notool"
      while [[ ! -s "${job}/COMPLETE.json" ]]; do
        [[ ! -s "${job}/FAILED.json" ]] || { cat "${job}/FAILED.json" >&2; exit 5; }
        sleep 30
      done
      if [[ ! -s "${job}/AUDIT.json" ]]; then
        PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
          "${PYTHON}" scripts/audit_sn7_controller_writeback_job.py \
            "${job}" "${job}/AUDIT.json" \
            --protocol-name sn7-controller-prior-corruption-v1 \
            --translation-pixels "${severity}" \
            --corruption-seed "${corruption_seed}" \
            >"${job}/audit.log" 2>&1
      fi
    done
  done
done

temporary="${RUN_ROOT}/NO_TOOL_AUDIT_COMPLETE.json.tmp.$$"
printf '{"status":"complete","jobs":12,"test_assets_read":false}\n' >"${temporary}"
mv -f "${temporary}" "${RUN_ROOT}/NO_TOOL_AUDIT_COMPLETE.json"
trap - EXIT
