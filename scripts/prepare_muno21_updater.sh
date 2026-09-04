#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
DATA_ROOT="${ACTIVEMAP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap}"
MUNO_ROOT="$DATA_ROOT/datasets/muno21/extracted/mapupdate"
OUTPUT="$DATA_ROOT/processed/muno21_v2/updater"
MANIFEST="$OUTPUT/updater_samples.jsonl"
AUDIT="$OUTPUT/audit.json"
QC_DIR="$OUTPUT/qc_train_val_v2"
READY="$OUTPUT/READY"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$QC_DIR"

if [[ ! -s "$MANIFEST" ]]; then
  "$PYTHON" -m activemap.cli build-updater-muno21 "$MUNO_ROOT" "$OUTPUT" \
    --image-size 512 --padding 128 --max-source-crop-size 1024 \
    --road-width-pixels 6 \
    --max-source-pixels 250000000 --seed 20260710
fi
"$PYTHON" -m activemap.cli audit-updater "$MANIFEST" "$AUDIT" \
  --max-reshape-centroid-distance 512 \
  --allow-empty-keep \
  --allow-nonlocal-polyline-reshape
"$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$AUDIT"

qc_count="$(find "$QC_DIR" -type f -name '*.png' | wc -l)"
if (( qc_count < 96 )); then
  "$PYTHON" -m activemap.cli render-updater-qc \
    "$MANIFEST" "$QC_DIR" --count 96 --seed 20260721
  qc_count="$(find "$QC_DIR" -type f -name '*.png' | wc -l)"
fi
if (( qc_count < 96 )); then
  echo "MUNO21 updater QC produced only $qc_count/96 PNG files" >&2
  exit 4
fi
printf '%s samples=%s qc=%s\n' \
  "$(date --iso-8601=seconds)" "$(wc -l < "$MANIFEST")" "$qc_count" >"$READY"
echo "[$(date --iso-8601=seconds)] MUNO21 updater preparation complete"
