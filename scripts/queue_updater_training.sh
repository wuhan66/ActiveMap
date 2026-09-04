#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${UPDATER_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/updater/base_v3_converged_seed20260710}"
GPU="${UPDATER_GPU:-0}"
POLL_SECONDS="${UPDATER_QUEUE_POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${UPDATER_QUEUE_MAX_WAIT_SECONDS:-86400}"
QUEUE_DIR="$RUN_DIR/queue"
QUEUE_PID_FILE="$QUEUE_DIR/queue.pid"
QUEUE_STATE="$QUEUE_DIR/state.txt"

mkdir -p "$QUEUE_DIR"
if [[ -f "$QUEUE_PID_FILE" ]] && kill -0 "$(cat "$QUEUE_PID_FILE")" 2>/dev/null; then
  echo "Updater queue already running: PID $(cat "$QUEUE_PID_FILE")" >&2
  exit 1
fi
echo "$$" > "$QUEUE_PID_FILE"
trap 'rm -f "$QUEUE_PID_FILE"' EXIT

started_at="$(date +%s)"
while true; do
  now="$(date +%s)"
  waited="$((now - started_at))"
  gpu_pids="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
  free_memory="$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)"
  printf '%s waiting_seconds=%s gpu=%s free_mib=%s compute_pids=%q\n' \
    "$(date --iso-8601=seconds)" "$waited" "$GPU" "$free_memory" "$gpu_pids" \
    > "$QUEUE_STATE"
  if [[ -z "${gpu_pids//[[:space:]]/}" ]] && (( free_memory >= 20000 )); then
    printf '%s launching updater on physical GPU %s\n' \
      "$(date --iso-8601=seconds)" "$GPU" >> "$QUEUE_STATE"
    UPDATER_GPU="$GPU" UPDATER_RUN_DIR="$RUN_DIR" \
      bash "$PROJECT_ROOT/scripts/start_updater_training.sh"
    exit 0
  fi
  if (( waited >= MAX_WAIT_SECONDS )); then
    printf '%s queue timeout after %s seconds\n' \
      "$(date --iso-8601=seconds)" "$waited" >> "$QUEUE_STATE"
    exit 2
  fi
  sleep "$POLL_SECONDS"
done
