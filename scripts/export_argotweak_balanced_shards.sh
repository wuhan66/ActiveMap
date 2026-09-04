#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
DATA_ROOT="${DATA_ROOT:-/home/wh/ActiveMap/datasets/argotweak/tbv_balanced_24_8_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/wh/ActiveMap/outputs/argotweak/balanced_24_8_v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/wh/ActiveMap/envs/argotweak-legacy/bin/python}"
OBJECT_THRESHOLD="${OBJECT_THRESHOLD:-0.3}"

shards=(
  "train_00_03:train_argotweak_00_03.pkl"
  "train_04_07:train_argotweak_04_07.pkl"
  "train_08_11:train_argotweak_08_11.pkl"
  "train_12_15:train_argotweak_12_15.pkl"
  "train_16_18:train_argotweak_16_18.pkl"
  "train_19:train_argotweak_19.pkl"
  "train_20_23:train_argotweak_20_23.pkl"
  "val_00_03:val_argotweak_00_03.pkl"
  "val_04_07:val_argotweak_04_07.pkl"
)

for shard_spec in "${shards[@]}"; do
  IFS=: read -r output_name annotation_name <<<"${shard_spec}"
  output_dir="${OUTPUT_ROOT}/${output_name}"
  "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/export_argotweak_official_proposals.py" \
    --results "${output_dir}/results.pkl" \
    --annotations "${DATA_ROOT}/official_shards/${annotation_name}" \
    --output "${output_dir}/atomic_edit_proposals.jsonl" \
    --object-threshold "${OBJECT_THRESHOLD}"
done
