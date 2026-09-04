#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-opencd/bin/python}"
LEDGER="${LEDGER:-${STORAGE_ROOT}/runs/frozen_test/sn7_safe_commit_20260727.ledger.json}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" scripts/run_frozen_paper_test.py \
  --purpose \
  "One-shot three-seed SN7 ChangeMamba/BAN safe-commit evaluation with AOI bootstrap" \
  --confirm-frozen \
  configs/experiments/paper_registry.yaml \
  "${STORAGE_ROOT}" \
  "${LEDGER}" \
  -- bash scripts/run_sn7_safe_commit_frozen_test.sh
