#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_uncertainty_v1/full_seed20260729}"

while [[ ! -f "${RUN_ROOT}/SUMMARY_COMPLETE.json" ]]; do sleep 60; done
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" scripts/build_sn7_controller_uncertainty_paper_assets.py \
    "${RUN_ROOT}/summary.json" "${RUN_ROOT}/paper_assets"
printf '{"status":"complete","test_assets_read":false}\n' \
  >"${RUN_ROOT}/PAPER_ASSETS_COMPLETE.json"
