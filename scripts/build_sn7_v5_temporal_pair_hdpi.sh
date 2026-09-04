#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
SOURCE_MANIFEST="${SN7_MANIFEST:-/home/wh/ActiveMap/manifests/sn7_frozen_v2/sn7_split.parquet}"
OUTPUT_DIR="${SN7_V5_OUTPUT:-/home/wh/ActiveMap/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ActiveMap Python runtime not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_MANIFEST" ]]; then
  echo "SN7 source manifest not found: $SOURCE_MANIFEST" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR/updater_samples.jsonl" || -e "$OUTPUT_DIR/summary.json" ]]; then
  echo "Refusing to overwrite an existing V5 data receipt: $OUTPUT_DIR" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
TRAINVAL_MANIFEST="$OUTPUT_DIR/sn7_trainval_source.parquet"
"$PYTHON" "$PROJECT_ROOT/scripts/materialize_trainval_manifest.py" \
  "$SOURCE_MANIFEST" "$TRAINVAL_MANIFEST"

"$PYTHON" -m activemap.cli build-updater-sn7 "$TRAINVAL_MANIFEST" "$OUTPUT_DIR" \
  --image-size 128 --context-pixels 32 --temporal-pair-input \
  --max-per-operation 20 --max-month-gap 1 --min-change-persistence 2 \
  --max-invalid-fraction 0.50 --min-area 16 --max-centroid-distance 20 \
  --seed 20260710
"$PYTHON" -m activemap.cli audit-updater \
  "$OUTPUT_DIR/updater_samples.jsonl" "$OUTPUT_DIR/audit.json"
"$PYTHON" -m activemap.cli render-updater-qc \
  "$OUTPUT_DIR/updater_samples.jsonl" "$OUTPUT_DIR/qc_train_val_v5" \
  --count 128 --splits train,val
