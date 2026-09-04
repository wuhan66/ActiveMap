#!/usr/bin/env bash
set -euo pipefail

# Render complete validation-only visual casebooks before qualitative curation.
# This job intentionally has no score-based example selection and never opens
# frozen-test assets.  SN7 requires fresh paired audits; MUNO21 and SpaceNet8
# consume immutable validation receipts.

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/runs/paper_visuals/full_validation_casebooks_20260904}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/full_validation_casebooks_20260904}"

MUNO_DIRECT="${MUNO_DIRECT:-${STORAGE_ROOT}/runs/agent/muno21_policy_baselines_hdpi_v2/writeback/generic_selector/writeback.jsonl}"
MUNO_ACTIVE="${MUNO_ACTIVE:-${STORAGE_ROOT}/runs/agent/muno21_policy_baselines_hdpi_v2/writeback/edit_conditioned_selector/writeback.jsonl}"
MUNO_ARRAYS="${MUNO_ARRAYS:-${STORAGE_ROOT}/processed/muno21_v2/updater/arrays}"
SN8_CANDIDATES="${SN8_CANDIDATES:-${STORAGE_ROOT}/runs/spacenet8_germany/active_multi_post_modern_cross_region_20260802}"
SN8_SELECTOR="${SN8_SELECTOR:-${SN8_CANDIDATES}/rank100_changer_selector_safe_commit}"

for path in "${PROJECT_ROOT}" "${PYTHON}" "${MUNO_DIRECT}" "${MUNO_ACTIVE}" "${MUNO_ARRAYS}" "${SN8_CANDIDATES}" "${SN8_SELECTOR}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

OUTPUT_ROOT="${OUTPUT_ROOT}/sn7" LOG_ROOT="${LOG_ROOT}/sn7" \
  bash scripts/run_sn7_latest_fullval_visuals_hdpi.sh >"${LOG_ROOT}/sn7.log" 2>&1 &
sn7_pid=$!

"${PYTHON}" scripts/render_muno21_full_validation_casebook.py \
  "${MUNO_DIRECT}" "${MUNO_ACTIVE}" "${MUNO_ARRAYS}" "${OUTPUT_ROOT}/muno21" \
  --tile-size 192 --zoom-size 256 >"${LOG_ROOT}/muno21.log" 2>&1 &
muno_pid=$!

"${PYTHON}" scripts/render_spacenet8_full_validation_casebook.py \
  "${SN8_CANDIDATES}" "${SN8_SELECTOR}" "${OUTPUT_ROOT}/spacenet8" \
  --tile-size 192 --zoom-size 256 >"${LOG_ROOT}/spacenet8.log" 2>&1 &
sn8_pid=$!

status=0
wait "${sn7_pid}" || status=1
wait "${muno_pid}" || status=1
wait "${sn8_pid}" || status=1
[[ "${status}" -eq 0 ]] || { echo "one or more casebook exports failed; inspect ${LOG_ROOT}" >&2; exit "${status}"; }

"${PYTHON}" - "${OUTPUT_ROOT}" "${MUNO_DIRECT}" "${MUNO_ACTIVE}" "${SN8_SELECTOR}" <<'PY'
import json
import sys
from pathlib import Path

root, muno_direct, muno_active, sn8_selector = map(Path, sys.argv[1:])
payload = {
    "schema_version": "full-validation-visual-casebooks-v1",
    "split": "val",
    "test_assets_read": False,
    "selection": "none; exhaustive export precedes qualitative curation",
    "domains": {
        "sn7": str(root / "sn7" / "bundle_summary.json"),
        "muno21": str(root / "muno21" / "summary.json"),
        "spacenet8": str(root / "spacenet8" / "summary.json"),
    },
    "sources": {
        "muno21_direct_writeback": str(muno_direct),
        "muno21_activemap_writeback": str(muno_active),
        "spacenet8_selector": str(sn8_selector),
    },
}
(root / "bundle_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
