#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_morphology_sensitivity_v1}"
MORPHOLOGY="${MORPHOLOGY:?set MORPHOLOGY}"
RADIUS="${RADIUS:?set RADIUS}"

case "${MORPHOLOGY}" in erode|dilate) ;; *) exit 2 ;; esac
case "${RADIUS}" in 2|8) ;; *) exit 2 ;; esac

cd "${PROJECT_ROOT}"
root="${RUN_ROOT}/${MORPHOLOGY}${RADIUS}"
if [[ ! -s "${root}/COMPLETE.json" ]]; then
  RUN_ROOT="${RUN_ROOT}" MORPHOLOGY="${MORPHOLOGY}" RADIUS="${RADIUS}" \
    bash scripts/run_sn7_controller_prior_morphology_states.sh
elif [[ ! -s "${root}/PARITY_AUDIT.json" ]]; then
  frozen="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/states_val_step0.jsonl"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}" \
      scripts/audit_sn7_frozen_anchor_parity.py \
      "${frozen}" "${root}/states_val_step0.jsonl" "${root}/PARITY_AUDIT.json"
fi

for seed in 20260730 20260731 20260801; do
  for variant in notool forced benefit; do
    RUN_ROOT="${RUN_ROOT}" MORPHOLOGY="${MORPHOLOGY}" RADIUS="${RADIUS}" \
      SEED="${seed}" VARIANT="${variant}" \
      bash scripts/run_sn7_controller_prior_morphology_job.sh
  done
done

printf \
  '{"status":"complete","morphology":"%s","radius":%s,"policy_seeds":3,"variants":3,"test_assets_read":false}\n' \
  "${MORPHOLOGY}" "${RADIUS}" >"${root}/MATRIX_COMPLETE.json"
