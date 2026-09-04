#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/server_hdpi_env.sh"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/acquire_training_slot.sh"
RUN_DIR="${OPERATION_SELECTOR_RUN_DIR:-${STORAGE_ROOT}/runs/selector/muno21_operation_v1_seed20260801}"
CONFIG="${OPERATION_SELECTOR_CONFIG:-$PROJECT_ROOT/configs/selector/muno21_operation_v1_server.yaml}"
PYTHON="${ACTIVEMAP_PYTHON:-${ACTIVEMAP_GIS_ENV}/bin/python}"
GPU="${OPERATION_SELECTOR_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
PID_FILE="$RUN_DIR/train.pid"

if [[ ! "$GPU" =~ ^(0|1)$ ]]; then
  echo "Refusing GPU $GPU: ActiveMap is restricted to physical GPUs 0 and 1" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "$STORAGE_ROOT"

mkdir -p "$RUN_DIR"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Operation selector training already running: PID $(cat "$PID_FILE")" >&2
  exit 1
fi
if [[ -f "$RUN_DIR/control/STOP" ]]; then
  echo "Remove $RUN_DIR/control/STOP before restarting training" >&2
  exit 1
fi
GPU_PIDS="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Physical GPU $GPU has active compute processes: $GPU_PIDS" >&2
  exit 1
fi

printf 'CUDA_VISIBLE_DEVICES=%q %q -m activemap.cli train-operation-selector %q --output %q\n' \
  "$GPU" "$PYTHON" "$CONFIG" "$RUN_DIR" > "$RUN_DIR/launch_command.txt"
nohup env CUDA_VISIBLE_DEVICES="$GPU" ACTIVEMAP_PYTHON="$PYTHON" \
  OPERATION_SELECTOR_CONFIG="$CONFIG" OPERATION_SELECTOR_RUN_DIR="$RUN_DIR" \
  bash "$PROJECT_ROOT/scripts/run_operation_selector_training_job.sh" \
  > "$RUN_DIR/train.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Operation selector training failed to start; inspect $RUN_DIR/train.log" >&2
  exit 1
fi
echo "Operation selector training started: PID $pid, physical GPU $GPU, run $RUN_DIR"
