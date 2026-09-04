#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/server_hdpi_env.sh"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/acquire_training_slot.sh"
RUN_DIR="${UPDATER_RUN_DIR:-${STORAGE_ROOT}/runs/updater/base_v3_converged_seed20260710}"
CONFIG="${UPDATER_CONFIG:-$PROJECT_ROOT/configs/updater/sn7_v3_server.yaml}"
PYTHON="${ACTIVEMAP_PYTHON:-${ACTIVEMAP_GIS_ENV}/bin/python}"
GPU="${UPDATER_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
ARCHIVE_DIR="${UPDATER_ARCHIVE_DIR:-$PROJECT_ROOT/outputs/updater/$(basename "$RUN_DIR")}"
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
  echo "Updater training already running: PID $(cat "$PID_FILE")" >&2
  exit 1
fi
if [[ -f "$RUN_DIR/control/STOP" ]]; then
  echo "Remove $RUN_DIR/control/STOP before restarting training" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for guarded GPU launch" >&2
  exit 1
fi
GPU_PIDS="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Physical GPU $GPU has active compute processes: $GPU_PIDS" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
printf 'CUDA_VISIBLE_DEVICES=%q %q -m activemap.cli train-updater %q --output %q\n' \
  "$GPU" "$PYTHON" "$CONFIG" "$RUN_DIR" > "$RUN_DIR/launch_command.txt"

nohup env CUDA_VISIBLE_DEVICES="$GPU" ACTIVEMAP_PYTHON="$PYTHON" \
  UPDATER_PHYSICAL_GPU="$GPU" \
  UPDATER_CONFIG="$CONFIG" UPDATER_RUN_DIR="$RUN_DIR" \
  UPDATER_ARCHIVE_DIR="$ARCHIVE_DIR" \
  bash "$PROJECT_ROOT/scripts/run_updater_training_job.sh" \
  > "$RUN_DIR/train.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Updater training failed to start; inspect $RUN_DIR/train.log" >&2
  exit 1
fi
echo "Updater training started: PID $pid, physical GPU $GPU, run $RUN_DIR"
