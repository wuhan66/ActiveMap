#!/usr/bin/env bash
set -euo pipefail

# Postprocess a completed exhaustive validation export.  It re-renders the
# small MUNO21 receipt casebook only to retain AOI metadata, then creates a
# deterministic qualitative selection registry after the SN7 casebook exists.

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
CASEBOOK_ROOT="${CASEBOOK_ROOT:-${STORAGE_ROOT}/runs/paper_visuals/full_validation_casebooks_20260904}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/full_validation_casebooks_20260904}"
MUNO_DIRECT="${MUNO_DIRECT:-${STORAGE_ROOT}/runs/agent/muno21_policy_baselines_hdpi_v2/writeback/generic_selector/writeback.jsonl}"
MUNO_ACTIVE="${MUNO_ACTIVE:-${STORAGE_ROOT}/runs/agent/muno21_policy_baselines_hdpi_v2/writeback/edit_conditioned_selector/writeback.jsonl}"
MUNO_ARRAYS="${MUNO_ARRAYS:-${STORAGE_ROOT}/processed/muno21_v2/updater/arrays}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${LOG_ROOT}"

if [[ ! -e "${CASEBOOK_ROOT}/muno21_aoi_v2" ]]; then
  "${PYTHON}" scripts/render_muno21_full_validation_casebook.py \
    "${MUNO_DIRECT}" "${MUNO_ACTIVE}" "${MUNO_ARRAYS}" "${CASEBOOK_ROOT}/muno21_aoi_v2" \
    --tile-size 192 --zoom-size 256 >"${LOG_ROOT}/muno21_aoi_v2.log" 2>&1
fi

while [[ ! -f "${CASEBOOK_ROOT}/sn7/full_casebook/summary.json" ]]; do
  sleep 90
done

if [[ ! -e "${CASEBOOK_ROOT}/qualitative_selection" ]]; then
  "${PYTHON}" scripts/build_qualitative_selection_registry.py \
    "${CASEBOOK_ROOT}/sn7/full_casebook/casebook_index.jsonl" \
    "${CASEBOOK_ROOT}/muno21_aoi_v2/casebook_index.jsonl" \
    "${CASEBOOK_ROOT}/spacenet8/casebook_index.jsonl" \
    "${CASEBOOK_ROOT}/qualitative_selection" \
    --max-per-stratum 1 >"${LOG_ROOT}/qualitative_selection.log" 2>&1
fi

"${PYTHON}" - "${CASEBOOK_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": "full-validation-visual-selection-v1",
    "split": "val",
    "test_assets_read": False,
    "source_casebooks": {
        "sn7": str(root / "sn7" / "full_casebook" / "summary.json"),
        "muno21": str(root / "muno21_aoi_v2" / "summary.json"),
        "spacenet8": str(root / "spacenet8" / "summary.json"),
    },
    "registry": str(root / "qualitative_selection" / "summary.json"),
}
(root / "selection_bundle_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
