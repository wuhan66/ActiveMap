#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
GPU="${MUNO21_ORACLE_GPU:-3}"
RUN_DIR="${MUNO21_ORACLE_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_oracle_v1}"
EPISODES="${MUNO21_EPISODES:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
OUTPUT="${MUNO21_ORACLE_OUTPUT:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/selector_states_v1.jsonl}"
UPDATER="${MUNO21_UPDATER_CHECKPOINT:-/mnt/mydisk/wh/ActiveMap/runs/updater/muno21_road_v4_explicit_change_scratch_seed20260726/best_val_loss.pt}"
OPERATION_SELECTOR="${MUNO21_OPERATION_SELECTOR_CHECKPOINT:-/mnt/mydisk/wh/ActiveMap/runs/selector/muno21_operation_v1_seed20260803/best.pt}"
COST_WEIGHT="${MUNO21_ORACLE_COST_WEIGHT:-0.02}"

mkdir -p "$RUN_DIR" "$(dirname "$OUTPUT")"
PID_FILE="$RUN_DIR/build.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "MUNO21 oracle already running: PID $(cat "$PID_FILE")" >&2
  exit 1
fi
GPU_PIDS="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Physical GPU $GPU has active compute processes: $GPU_PIDS" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
printf 'physical_gpu=%s\noutput=%s\ncost_weight=%s\n' \
  "$GPU" "$OUTPUT" "$COST_WEIGHT" > "$RUN_DIR/launch.txt"
nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PROJECT_ROOT/src" \
  "$PYTHON" -m activemap.cli build-selector-oracle \
  "$UPDATER" "$EPISODES" "$OUTPUT" \
  --device cuda:0 --image-size 512 --budgets 1.5,3.0,4.5 --cost-weight "$COST_WEIGHT" \
  --operation-selector-checkpoint "$OPERATION_SELECTOR" \
  --operation-update-threshold 0.69 \
  > "$RUN_DIR/build.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "MUNO21 oracle failed to start; inspect $RUN_DIR/build.log" >&2
  exit 1
fi
echo "MUNO21 oracle started: PID $pid, physical GPU $GPU"
