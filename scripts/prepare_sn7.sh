#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-python}"
RAW_ROOT="${1:?Usage: bash scripts/prepare_sn7.sh /path/to/extracted/sn7 [processed_dir] [manifest_dir] [split_dir]}"
PROCESSED_DIR="${2:-$ROOT_DIR/data/processed/sn7_v1}"
MANIFEST_DIR="${3:-$ROOT_DIR/data/manifests}"
SPLIT_DIR="${4:-$ROOT_DIR/splits}"

mkdir -p "$MANIFEST_DIR" "$PROCESSED_DIR" "$SPLIT_DIR"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m activemap index-sn7 "$RAW_ROOT" "$MANIFEST_DIR/sn7.parquet"
"$PYTHON" -m activemap make-splits "$MANIFEST_DIR/sn7.parquet" "$SPLIT_DIR" \
  --manifest-output "$MANIFEST_DIR/sn7_split.parquet" --seed 20260710
"$PYTHON" -m activemap audit-manifest "$MANIFEST_DIR/sn7_split.parquet"
"$PYTHON" -m activemap scan-sn7-edits "$MANIFEST_DIR/sn7_split.parquet" \
  "$PROCESSED_DIR/pair_counts.parquet" --max-per-operation 5 \
  --max-month-gap 1 --min-change-persistence 2 --min-area 16 \
  --max-centroid-distance 20 --seed 20260710
"$PYTHON" -m activemap build-updater-sn7 "$MANIFEST_DIR/sn7_split.parquet" \
  "$PROCESSED_DIR/updater" --image-size 128 --context-pixels 32 \
  --max-per-operation 5 --max-month-gap 1 --min-change-persistence 2 \
  --max-invalid-fraction 0.50 --min-area 16 --max-centroid-distance 20 \
  --seed 20260710
"$PYTHON" -m activemap audit-updater \
  "$PROCESSED_DIR/updater/updater_samples.jsonl" \
  "$PROCESSED_DIR/updater/audit.json"
"$PYTHON" -m activemap build-episodes-sn7 "$MANIFEST_DIR/sn7_split.parquet" \
  "$PROCESSED_DIR/episodes.jsonl" --max-per-operation 5 \
  --max-month-gap 1 --min-change-persistence 2 --min-area 16 \
  --max-centroid-distance 20 --seed 20260710
"$PYTHON" -m activemap audit-episodes "$PROCESSED_DIR/episodes.jsonl" \
  "$PROCESSED_DIR/episodes.audit.json" \
  --expected-derivation-version sn7-adjacent-v3-distance-gated
"$PYTHON" -m activemap render-updater-qc \
  "$PROCESSED_DIR/updater/updater_samples.jsonl" "$PROCESSED_DIR/qc" --count 64
