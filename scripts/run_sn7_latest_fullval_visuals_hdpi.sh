#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
MANIFEST="${MANIFEST:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1/updater_samples.jsonl}"
LATEST_CHECKPOINT="${LATEST_CHECKPOINT:-${STORAGE_ROOT}/runs/updater/sn7_carried_runtime_temporal_seed20260903/best_quality.pt}"
FROZEN_CHECKPOINT="${FROZEN_CHECKPOINT:-${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed20260817/best_quality.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE_ROOT}/runs/paper_visuals/sn7_full_validation_casebook_20260904}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/sn7_full_validation_casebook_20260904}"
LATEST_GPU="${LATEST_GPU:-3}"
FROZEN_GPU="${FROZEN_GPU:-4}"

for path in "${PYTHON}" "${MANIFEST}" "${LATEST_CHECKPOINT}" "${FROZEN_CHECKPOINT}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "refusing to overwrite visual run: ${OUTPUT_ROOT}" >&2
  exit 2
}

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${LATEST_GPU}" "${PYTHON}" \
  scripts/evaluate_sn7_updater_visual_audit.py \
  "${LATEST_CHECKPOINT}" "${MANIFEST}" "${OUTPUT_ROOT}/latest_audit" \
  --split val --device cuda:0 --batch-size 12 --workers 8 --prefetch-factor 4 \
  --input-size 512 \
  >"${LOG_ROOT}/latest_audit.log" 2>&1 &
latest_pid=$!

CUDA_VISIBLE_DEVICES="${FROZEN_GPU}" "${PYTHON}" \
  scripts/evaluate_sn7_updater_visual_audit.py \
  "${FROZEN_CHECKPOINT}" "${MANIFEST}" "${OUTPUT_ROOT}/frozen_audit" \
  --split val --device cuda:0 --batch-size 12 --workers 8 --prefetch-factor 4 \
  >"${LOG_ROOT}/frozen_audit.log" 2>&1 &
frozen_pid=$!

status=0
wait "${latest_pid}" || status=1
wait "${frozen_pid}" || status=1
[[ "${status}" -eq 0 ]] || {
  echo "one or both validation audits failed; inspect ${LOG_ROOT}" >&2
  exit "${status}"
}

"${PYTHON}" scripts/render_sn7_full_validation_atlas.py \
  "${MANIFEST}" frozen "${OUTPUT_ROOT}/frozen_audit" \
  latest "${OUTPUT_ROOT}/latest_audit" "${OUTPUT_ROOT}/full_atlas" \
  --columns 8 --rows 8 --tile-size 96 \
  >"${LOG_ROOT}/full_atlas.log" 2>&1

"${PYTHON}" scripts/render_sn7_full_validation_casebook.py \
  "${MANIFEST}" "${OUTPUT_ROOT}/full_casebook" \
  --audit "frozen=${OUTPUT_ROOT}/frozen_audit" \
  --audit "latest=${OUTPUT_ROOT}/latest_audit" \
  --tile-size 192 \
  >"${LOG_ROOT}/full_casebook.log" 2>&1

"${PYTHON}" - "${OUTPUT_ROOT}" "${MANIFEST}" "${LATEST_CHECKPOINT}" "${FROZEN_CHECKPOINT}" <<'PY'
import json
import sys
from pathlib import Path

output, manifest, latest, frozen = (Path(value) for value in sys.argv[1:])
payload = {
    "schema_version": "sn7-full-validation-visual-bundle-v2",
    "split": "val",
    "manifest": str(manifest),
    "latest_checkpoint": str(latest),
    "latest_checkpoint_status": "diagnostic_only_not_a_paper_claim",
    "frozen_checkpoint": str(frozen),
    "latest_audit": str(output / "latest_audit" / "summary.json"),
    "frozen_audit": str(output / "frozen_audit" / "summary.json"),
    "full_atlas": str(output / "full_atlas" / "summary.json"),
    "full_casebook": str(output / "full_casebook" / "summary.json"),
    "selection": "none; exhaustive validation export before figure curation",
    "test_assets_read": False,
}
(output / "bundle_summary.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
