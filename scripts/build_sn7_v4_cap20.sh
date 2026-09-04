#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
MANIFEST="${SN7_MANIFEST:-/mnt/mydisk/wh/ActiveMap/manifests/sn7_split.parquet}"
OUTPUT_DIR="${SN7_V4_OUTPUT:-/mnt/mydisk/wh/ActiveMap/processed/sn7_v1/updater_v4_cap20}"
MAX_PER_OPERATION="${SN7_V4_MAX_PER_OPERATION:-20}"

if [[ "$OUTPUT_DIR" == *"updater_v3"* ]]; then
  echo "Refusing to write v4 data into a v3 directory: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

printf '%s build started manifest=%s output=%s max_per_operation=%s\n' \
  "$(date --iso-8601=seconds)" "$MANIFEST" "$OUTPUT_DIR" "$MAX_PER_OPERATION"
"$PYTHON" -m activemap.cli build-updater-sn7 "$MANIFEST" "$OUTPUT_DIR" \
  --image-size 128 --context-pixels 32 \
  --max-per-operation "$MAX_PER_OPERATION" \
  --max-month-gap 1 --min-change-persistence 2 \
  --max-invalid-fraction 0.50 --min-area 16 --max-centroid-distance 20 \
  --seed 20260710
"$PYTHON" -m activemap.cli audit-updater \
  "$OUTPUT_DIR/updater_samples.jsonl" "$OUTPUT_DIR/audit.json"
"$PYTHON" -m activemap.cli render-updater-qc \
  "$OUTPUT_DIR/updater_samples.jsonl" "$OUTPUT_DIR/qc_train_val_v2" \
  --count 128 --splits train,val
printf '%s build and audit completed\n' "$(date --iso-8601=seconds)"
