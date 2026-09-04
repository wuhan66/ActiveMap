#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${ACTIVEMAP_GIS_ENV:-/home/wh/venvs/activemap}/bin/python}"
QC_NAME="qc_train_val_v2"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

render_qc() {
  local dataset="$1"
  local manifest="$2"
  local output="$3"
  local count="$4"
  local seed="$5"
  [[ -s "${manifest}" ]] || { echo "missing ${dataset} manifest: ${manifest}" >&2; exit 2; }
  [[ ! -e "${output}" ]] || {
    echo "refusing existing ${dataset} paper QC directory: ${output}" >&2
    exit 3
  }
  "${PYTHON}" -m activemap.cli render-updater-qc \
    "${manifest}" "${output}" --count "${count}" --seed "${seed}" \
    --splits train,val
  "${PYTHON}" - "${dataset}" "${output}/index.json" "${count}" <<'PY'
import json
import pathlib
import sys

dataset, path, expected = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
index = json.loads(path.read_text())
assert index["rendered"] == expected, (dataset, index["rendered"], expected)
assert index["test_assets_rendered"] is False, dataset
assert set(index["rendered_split_counts"]) <= {"train", "val"}, dataset
assert len(list(path.parent.glob("*.png"))) == expected, dataset
PY
}

render_qc sn7 \
  "${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl" \
  "${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/${QC_NAME}" 128 20260710
render_qc muno21 \
  "${STORAGE_ROOT}/processed/muno21_v2/updater/updater_samples.jsonl" \
  "${STORAGE_ROOT}/processed/muno21_v2/updater/${QC_NAME}" 96 20260721
render_qc inria \
  "${STORAGE_ROOT}/processed/inria_v1/segmentation/updater_samples.jsonl" \
  "${STORAGE_ROOT}/processed/inria_v1/segmentation/${QC_NAME}" 96 20260721

echo "[$(date --iso-8601=seconds)] paper train/validation QC rendered"
