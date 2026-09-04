#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "$PROJECT_ROOT/scripts/server_hdpi_env.sh"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_PYTHON:-${ACTIVEMAP_GIS_ENV}/bin/python}"
GPU="${MUNO21_SELECTOR_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
CONFIG="${MUNO21_SELECTOR_CONFIG:-$PROJECT_ROOT/configs/selector/muno21_evidence_v1_server.yaml}"
RUN_DIR="${MUNO21_SELECTOR_RUN_DIR:-${STORAGE_ROOT}/runs/selector/muno21_evidence_v1_seed20260811}"
STATES="${MUNO21_SELECTOR_STATES:-${STORAGE_ROOT}/processed/muno21_v2/agent/selector_states_v1.jsonl}"

if [[ ! "$GPU" =~ ^(0|1)$ ]]; then
  echo "Refusing GPU $GPU: ActiveMap is restricted to physical GPUs 0 and 1" >&2
  exit 2
fi
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "$STORAGE_ROOT"

if [[ ! -s "$STATES" || ! -s "${STATES%.jsonl}.summary.json" ]]; then
  echo "Selector states are incomplete: $STATES" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
PID_FILE="$RUN_DIR/train.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Evidence selector already running: PID $(cat "$PID_FILE")" >&2
  exit 1
fi
GPU_PIDS="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Physical GPU $GPU has active compute processes: $GPU_PIDS" >&2
  exit 1
fi

printf 'physical_gpu=%s\nconfig=%s\nrun_dir=%s\n' "$GPU" "$CONFIG" "$RUN_DIR" \
  > "$RUN_DIR/launch.txt"
nohup env CUDA_VISIBLE_DEVICES="$GPU" ACTIVEMAP_PYTHON="$PYTHON" \
  MUNO21_SELECTOR_CONFIG="$CONFIG" MUNO21_SELECTOR_RUN_DIR="$RUN_DIR" \
  bash "$PROJECT_ROOT/scripts/run_muno21_evidence_selector_job.sh" \
  > "$RUN_DIR/train.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Evidence selector failed to start; inspect $RUN_DIR/train.log" >&2
  exit 1
fi
echo "Evidence selector started: PID $pid, physical GPU $GPU, run $RUN_DIR"
