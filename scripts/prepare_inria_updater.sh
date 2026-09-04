#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
DATA_ROOT="${ACTIVEMAP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap}"
INRIA_ROOT="$DATA_ROOT/datasets/inria_aerial/extracted"
INRIA_OUTPUT="$DATA_ROOT/processed/inria_v1/updater"
MANIFEST="$INRIA_OUTPUT/updater_samples.jsonl"
AUDIT="$INRIA_OUTPUT/audit.json"
READY="$INRIA_OUTPUT/READY"
QC_DIR="$INRIA_OUTPUT/qc"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$QC_DIR"

if [[ -s "$MANIFEST" && -s "$AUDIT" ]]; then
  "$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$AUDIT"
  qc_count="$(find "$QC_DIR" -type f -name '*.png' 2>/dev/null | wc -l)"
  if (( qc_count < 96 )); then
    echo "[$(date --iso-8601=seconds)] audit passed but QC is incomplete ($qc_count/96); rendering"
    "$PYTHON" -m activemap.cli render-updater-qc \
      "$MANIFEST" "$QC_DIR" --count 96 --seed 20260721
    qc_count="$(find "$QC_DIR" -type f -name '*.png' 2>/dev/null | wc -l)"
  fi
  if (( qc_count < 96 )); then
    echo "Inria QC rendering produced only $qc_count/96 PNG files" >&2
    exit 4
  fi
  touch "$READY"
  echo "[$(date --iso-8601=seconds)] audited Inria manifest and complete QC already exist; skipping"
  exit 0
fi

if ! find "$INRIA_ROOT" -type f -path '*/train/images/*.tif' -print -quit | grep -q .; then
  echo "no extracted Inria train/images GeoTIFFs under $INRIA_ROOT" >&2
  exit 3
fi
if ! find "$INRIA_ROOT" -type f -path '*/train/gt/*.tif' -print -quit | grep -q .; then
  echo "no extracted Inria train/gt GeoTIFFs under $INRIA_ROOT" >&2
  exit 3
fi

"$PYTHON" -m activemap.cli build-updater-inria "$INRIA_ROOT" "$INRIA_OUTPUT" \
  --image-size 256 --context-pixels 48 --max-objects-per-tile 64 \
  --min-area-pixels 16 --min-valid-fraction 0.5 \
  --validation-cities vienna --test-cities kitsap \
  --seed 20260710
"$PYTHON" -m activemap.cli audit-updater "$MANIFEST" "$AUDIT"
"$PYTHON" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$AUDIT"
"$PYTHON" -m activemap.cli render-updater-qc \
  "$MANIFEST" "$QC_DIR" --count 96 --seed 20260721
qc_count="$(find "$QC_DIR" -type f -name '*.png' 2>/dev/null | wc -l)"
if (( qc_count < 96 )); then
  echo "Inria QC rendering produced only $qc_count/96 PNG files" >&2
  exit 4
fi
touch "$READY"
echo "[$(date --iso-8601=seconds)] Inria updater preparation complete"
