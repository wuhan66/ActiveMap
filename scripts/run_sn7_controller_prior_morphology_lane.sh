#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_v1}"
POLICY_SEED="${POLICY_SEED:?set POLICY_SEED}"
POLL_SECONDS="${POLL_SECONDS:-30}"

case "${POLICY_SEED}" in 20260730|20260731|20260801) ;; *) exit 2 ;; esac

cd "${PROJECT_ROOT}"
for specification in erode:4 dilate:4; do
  IFS=: read -r morphology radius <<<"${specification}"
  root="${RUN_ROOT}/${morphology}${radius}"
  failure="${root}/FAILED.json"
  while [[ ! -s "${root}/COMPLETE.json" || ! -s "${root}/PARITY_AUDIT.json" ]]; do
    [[ ! -s "${failure}" ]] || { cat "${failure}" >&2; exit 5; }
    sleep "${POLL_SECONDS}"
  done
  for variant in notool forced benefit; do
    RUN_ROOT="${RUN_ROOT}" MORPHOLOGY="${morphology}" RADIUS="${radius}" \
      SEED="${POLICY_SEED}" VARIANT="${variant}" \
      bash scripts/run_sn7_controller_prior_morphology_job.sh
  done
done
