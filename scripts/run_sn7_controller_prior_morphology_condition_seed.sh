#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_sensitivity_v1}"
MORPHOLOGY="${MORPHOLOGY:?set MORPHOLOGY}"
RADIUS="${RADIUS:?set RADIUS}"
SEED="${SEED:?set SEED}"
POLL_SECONDS="${POLL_SECONDS:-30}"

case "${MORPHOLOGY}" in erode|dilate) ;; *) exit 2 ;; esac
case "${RADIUS}" in 2|8) ;; *) exit 2 ;; esac
case "${SEED}" in 20260731|20260801) ;; *) exit 2 ;; esac

root="${RUN_ROOT}/${MORPHOLOGY}${RADIUS}"
while [[ ! -s "${root}/COMPLETE.json" || ! -s "${root}/PARITY_AUDIT.json" ]]; do
  [[ ! -s "${root}/FAILED.json" ]] || exit 5
  sleep "${POLL_SECONDS}"
done

cd "${PROJECT_ROOT}"
for variant in notool forced benefit; do
  RUN_ROOT="${RUN_ROOT}" MORPHOLOGY="${MORPHOLOGY}" RADIUS="${RADIUS}" \
    SEED="${SEED}" VARIANT="${variant}" \
    bash scripts/run_sn7_controller_prior_morphology_job.sh
done
