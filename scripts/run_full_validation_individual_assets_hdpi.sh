#!/usr/bin/env bash
set -euo pipefail

# Re-render all existing validation evidence as independent assets.  This does
# not run inference: it consumes immutable validation audits/receipts and
# emits unlabelled semantic layers for author-side composition.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-${STORAGE_ROOT}/runs/paper_visuals/full_validation_casebooks_20260904}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/runs/paper_visuals/full_validation_individual_assets_20260904}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/full_validation_individual_assets_20260904}"
SN7_MANIFEST="${SN7_MANIFEST:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1/updater_samples.jsonl}"
MUNO_DIRECT="${MUNO_DIRECT:-${STORAGE_ROOT}/runs/agent/muno21_policy_baselines_hdpi_v2/writeback/generic_selector/writeback.jsonl}"
MUNO_ACTIVE="${MUNO_ACTIVE:-${STORAGE_ROOT}/runs/agent/muno21_policy_baselines_hdpi_v2/writeback/edit_conditioned_selector/writeback.jsonl}"
MUNO_ARRAYS="${MUNO_ARRAYS:-${STORAGE_ROOT}/processed/muno21_v2/updater/arrays}"
SN8_CANDIDATES="${SN8_CANDIDATES:-${STORAGE_ROOT}/runs/spacenet8_germany/active_multi_post_modern_cross_region_20260802}"
SN8_SELECTOR="${SN8_SELECTOR:-${SN8_CANDIDATES}/rank100_changer_selector_safe_commit}"

for path in "${PROJECT_ROOT}" "${PYTHON}" "${SN7_MANIFEST}" \
  "${SOURCE_ROOT}/sn7/frozen_audit" "${SOURCE_ROOT}/sn7/latest_audit" \
  "${MUNO_DIRECT}" "${MUNO_ACTIVE}" "${MUNO_ARRAYS}" "${SN8_CANDIDATES}" "${SN8_SELECTOR}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 2; }
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" scripts/render_sn7_full_validation_casebook.py \
  "${SN7_MANIFEST}" "${OUTPUT_ROOT}/sn7" \
  --audit "frozen=${SOURCE_ROOT}/sn7/frozen_audit" \
  --audit "latest=${SOURCE_ROOT}/sn7/latest_audit" \
  --asset-mode individual >"${LOG_ROOT}/sn7.log" 2>&1 &
sn7_pid=$!

"${PYTHON}" scripts/render_muno21_full_validation_casebook.py \
  "${MUNO_DIRECT}" "${MUNO_ACTIVE}" "${MUNO_ARRAYS}" "${OUTPUT_ROOT}/muno21" \
  --asset-mode individual >"${LOG_ROOT}/muno21.log" 2>&1 &
muno_pid=$!

"${PYTHON}" scripts/render_spacenet8_full_validation_casebook.py \
  "${SN8_CANDIDATES}" "${SN8_SELECTOR}" "${OUTPUT_ROOT}/spacenet8" \
  --asset-mode individual >"${LOG_ROOT}/spacenet8.log" 2>&1 &
sn8_pid=$!

status=0
wait "${sn7_pid}" || status=1
wait "${muno_pid}" || status=1
wait "${sn8_pid}" || status=1
[[ "${status}" -eq 0 ]] || { echo "one or more individual asset exports failed" >&2; exit "${status}"; }

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
payload = {
    "schema_version": "full-validation-individual-assets-v1",
    "split": "val",
    "test_assets_read": False,
    "asset_mode": "individual",
    "composition": "none; each source image and semantic layer is stored separately",
    "domains": {name: str(root / name / "summary.json") for name in ("sn7", "muno21", "spacenet8")},
}
(root / "bundle_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
