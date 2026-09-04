#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1/full_seed20260729}"

while [[ ! -f "${RUN_ROOT}/MATRIX_COMPLETE.json" ]]; do sleep 60; done
cd "${PROJECT_ROOT}"

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" scripts/audit_sn7_controller_tool_independence.py \
    "${RUN_ROOT}" --output "${RUN_ROOT}/tool_independence_audit.json"

printf '{"status":"complete","test_assets_read":false}\n' \
  >"${RUN_ROOT}/TOOL_AUDIT_COMPLETE.json"
