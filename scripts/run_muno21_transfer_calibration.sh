#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
DATA_ROOT="${ACTIVEMAP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap}"
GPU="${GPU:-3}"
SAMPLES="${MUNO21_SAMPLES:-${DATA_ROOT}/processed/muno21_v2/updater/updater_samples.jsonl}"
SCRATCH_RUN="${SCRATCH_RUN:-${DATA_ROOT}/runs/updater/muno21_road_v7_topology_scratch_seed20260731}"
TRANSFER_RUN="${TRANSFER_RUN:-${DATA_ROOT}/runs/updater/muno21_road_v8_from_sn7_encoder_seed20260731}"
OUTPUT_DIR="${OUTPUT_DIR:-${TRANSFER_RUN}/calibrated_comparison_vs_scratch}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for path in "$PYTHON" "$SAMPLES" "$SCRATCH_RUN/best_quality.pt" "$TRANSFER_RUN/best_quality.pt"; do
  if [[ ! -e "$path" ]]; then
    echo "Required calibration input is missing: $path" >&2
    exit 2
  fi
done
if pgrep -u "$USER" -f '[a]ctivemap.cli train-updater|[t]rain_tool_belief_spatial_decision_head.py|[t]rain_agent_(sft|dpo).py' >/dev/null; then
  echo "Another ActiveMap training experiment is running" >&2
  exit 3
fi
gpu_uuid="$({ nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits || true; } \
  | awk -F, -v gpu_index="$GPU" \
    '$1 + 0 == gpu_index {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
if [[ -z "$gpu_uuid" ]]; then
  echo "Physical GPU $GPU was not found" >&2
  exit 4
fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits \
  | grep -Fxq "$gpu_uuid"; then
  echo "Physical GPU $GPU is not free; refusing calibration" >&2
  exit 5
fi

mkdir -p "$OUTPUT_DIR"
for name in scratch transfer; do
  if [[ "$name" == "scratch" ]]; then
    checkpoint="$SCRATCH_RUN/best_quality.pt"
  else
    checkpoint="$TRANSFER_RUN/best_quality.pt"
  fi
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -m activemap.cli \
    calibrate-updater-temporal-change "$checkpoint" "$SAMPLES" \
    "$OUTPUT_DIR/${name}_calibration.json" --split val --device cuda:0 \
    --batch-size 4 --num-workers 4 --max-stable-false-positive 0.005 \
    --grid-steps 37
done

"$PYTHON" scripts/compare_temporal_calibrations.py \
  --run "scratch=$OUTPUT_DIR/scratch_calibration.json" \
  --run "transfer=$OUTPUT_DIR/transfer_calibration.json" \
  --output "$OUTPUT_DIR/summary.json"
