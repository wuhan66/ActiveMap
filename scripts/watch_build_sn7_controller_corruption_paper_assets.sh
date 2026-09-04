#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
BASE="${STORAGE_ROOT}/runs/sn7_active_catalog/controller_prior_corruption_v1"
PRIMARY="${BASE}/full_seed20260729"
SENSITIVITY="${BASE}/severe_seed_sensitivity_v1"
OUTPUT="${PRIMARY}/paper_assets"

for marker in \
  "${PRIMARY}/SUMMARY_COMPLETE.json" \
  "${PRIMARY}/TOOL_AUDIT_COMPLETE.json" \
  "${SENSITIVITY}/COMPLETE.json"; do
  while [[ ! -f "${marker}" ]]; do sleep 60; done
done

cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" scripts/build_sn7_controller_corruption_paper_assets.py \
    "${PRIMARY}/summary.json" \
    "${SENSITIVITY}/benefit_vs_notool.json" \
    "${PRIMARY}/tool_independence_audit.json" \
    "${OUTPUT}"

printf '{"status":"complete","test_assets_read":false}\n' \
  >"${PRIMARY}/PAPER_ASSETS_COMPLETE.json"
