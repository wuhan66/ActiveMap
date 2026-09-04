#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1}"
mkdir -p "${RUN_ROOT}"
FAILURE_MARKER="${RUN_ROOT}/MATRIX_FAILED.json"
rm -f "${FAILURE_MARKER}"
record_failure() {
  local rc=$?
  trap - EXIT
  if ((rc != 0)); then
    local temporary="${FAILURE_MARKER}.tmp.$$"
    printf \
      '{"status":"failed","stage":"morphology_matrix","exit_code":%d,"test_assets_read":false}\n' \
      "${rc}" >"${temporary}"
    mv -f "${temporary}" "${FAILURE_MARKER}"
  fi
  exit "${rc}"
}
trap record_failure EXIT

cd "${PROJECT_ROOT}"
for specification in erode:4 dilate:4; do
  IFS=: read -r morphology radius <<<"${specification}"
  root="${RUN_ROOT}/${morphology}${radius}"
  if [[ ! -s "${root}/COMPLETE.json" ]]; then
    RUN_ROOT="${RUN_ROOT}" MORPHOLOGY="${morphology}" RADIUS="${radius}" \
      bash scripts/run_sn7_controller_prior_morphology_states.sh
  fi
  for seed in 20260730 20260731 20260801; do
    for variant in notool forced benefit; do
      marker="${root}/seed${seed}/${variant}/COMPLETE.json"
      [[ -s "${marker}" ]] && continue
      RUN_ROOT="${RUN_ROOT}" MORPHOLOGY="${morphology}" RADIUS="${radius}" \
        SEED="${seed}" VARIANT="${variant}" \
        bash scripts/run_sn7_controller_prior_morphology_job.sh
    done
  done
  printf \
    '{"status":"complete","morphology":"%s","radius":%s,"policy_seeds":3,"variants":3,"test_assets_read":false}\n' \
    "${morphology}" "${radius}" >"${root}/MATRIX_COMPLETE.json"
done
trap - EXIT
