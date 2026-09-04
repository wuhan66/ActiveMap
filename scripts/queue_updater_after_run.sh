#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
PREREQUISITE_RUN="${PREREQUISITE_RUN:?PREREQUISITE_RUN is required}"
TARGET_CONFIG="${TARGET_CONFIG:?TARGET_CONFIG is required}"
TARGET_RUN="${TARGET_RUN:?TARGET_RUN is required}"
TARGET_ARCHIVE="${TARGET_ARCHIVE:-$PROJECT_ROOT/outputs/updater/$(basename "$TARGET_RUN")}"
GPU="${UPDATER_GPU:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"
QUEUE_PID_FILE="${QUEUE_PID_FILE:-}"

if [[ -n "$QUEUE_PID_FILE" ]]; then
  mkdir -p "$(dirname "$QUEUE_PID_FILE")"
  echo "$$" > "$QUEUE_PID_FILE"
fi

for path in "$PYTHON" "$PREREQUISITE_RUN/state.json" "$TARGET_CONFIG"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path is missing: $path" >&2
    exit 2
  fi
done
if [[ -e "$TARGET_RUN" ]]; then
  echo "Refusing to overwrite target run: $TARGET_RUN" >&2
  exit 3
fi

while true; do
  status="$($PYTHON -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' \
    "$PREREQUISITE_RUN/state.json")"
  printf '%s prerequisite_status=%s\n' "$(date --iso-8601=seconds)" "$status"
  if [[ "$status" == "completed" ]]; then
    break
  fi
  if [[ "$status" == "failed" || "$status" == "stopped" ]]; then
    echo "Prerequisite did not complete successfully; target will not start" >&2
    exit 4
  fi
  sleep "$POLL_SECONDS"
done

if [[ ! -f "$PREREQUISITE_RUN/best_safety.pt" ]]; then
  echo "Completed prerequisite has no hard-gated best_safety.pt" >&2
  exit 5
fi

while true; do
  gpu_pids="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -z "${gpu_pids//[[:space:]]/}" ]]; then
    break
  fi
  printf '%s waiting_for_gpu=%s active_pids=%s\n' \
    "$(date --iso-8601=seconds)" "$GPU" "$(echo "$gpu_pids" | tr '\n' ',')"
  sleep 10
done

cd "$PROJECT_ROOT"
UPDATER_CONFIG="$TARGET_CONFIG" \
UPDATER_RUN_DIR="$TARGET_RUN" \
UPDATER_ARCHIVE_DIR="$TARGET_ARCHIVE" \
UPDATER_GPU="$GPU" \
  bash scripts/start_updater_training.sh
