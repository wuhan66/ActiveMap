#!/usr/bin/env bash
set -euo pipefail

# Exhaustive validation-only HD-map boards. This is CPU rendering and does not
# alter frozen perception outputs or require a GPU allocation.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/runs/argotweak/native_adapter_frozen_baseline_v2/episodes/val.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${STORAGE_ROOT}/runs/paper_visuals/argotweak_validation_casebook_20260904}"
LOG_DIR="${LOG_DIR:-${STORAGE_ROOT}/logs/argotweak_validation_casebook_20260904}"
ASSET_MODE="${ASSET_MODE:-board}"

for path in "${PROJECT_ROOT}" "${PYTHON}" "${EPISODES}"; do
  [[ -e "${path}" ]] || { echo "missing required input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT_DIR}" ]] || { echo "refusing to overwrite ${OUTPUT_DIR}" >&2; exit 2; }
mkdir -p "${LOG_DIR}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" scripts/render_argotweak_full_validation_casebook.py "${EPISODES}" "${OUTPUT_DIR}" \
  --camera-width 224 --camera-height 144 --map-size 448 --overview-radius 90 --crop-radius 35 \
  --asset-mode "${ASSET_MODE}" \
  >"${LOG_DIR}/render.log" 2>&1
