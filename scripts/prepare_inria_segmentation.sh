#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
DATA_ROOT="${ACTIVEMAP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap}"
INRIA_ROOT="$DATA_ROOT/datasets/inria_aerial/extracted"
OUTPUT="$DATA_ROOT/processed/inria_v1/segmentation"
MANIFEST="$OUTPUT/updater_samples.jsonl"
AUDIT="$OUTPUT/audit.json"
QC_DIR="$OUTPUT/qc_train_val_v2"
READY="$OUTPUT/READY"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$QC_DIR"

if [[ ! -s "$MANIFEST" ]]; then
  "$PYTHON" -m activemap.cli build-inria-segmentation "$INRIA_ROOT" "$OUTPUT" \
    --image-size 256 --window-size 512 --stride 512 --min-valid-fraction 0.5 \
    --validation-cities vienna --test-cities kitsap --seed 20260721
fi
"$PYTHON" -m activemap.cli audit-updater "$MANIFEST" "$AUDIT" --allow-empty-keep
"$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$AUDIT"

qc_count="$(find "$QC_DIR" -type f -name '*.png' | wc -l)"
if (( qc_count < 96 )); then
  "$PYTHON" -m activemap.cli render-updater-qc \
    "$MANIFEST" "$QC_DIR" --count 96 --seed 20260721
  qc_count="$(find "$QC_DIR" -type f -name '*.png' | wc -l)"
fi
if (( qc_count < 96 )); then
  echo "Inria segmentation QC produced only $qc_count/96 PNG files" >&2
  exit 4
fi
touch "$READY"
echo "[$(date --iso-8601=seconds)] Inria semantic segmentation preparation complete"
