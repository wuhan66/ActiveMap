#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${UPDATER_RUN_DIR:?UPDATER_RUN_DIR is required}"
CONFIG="${UPDATER_CONFIG:?UPDATER_CONFIG is required}"
PYTHON="${ACTIVEMAP_PYTHON:?ACTIVEMAP_PYTHON is required}"
ARCHIVE_DIR="${UPDATER_ARCHIVE_DIR:-$PROJECT_ROOT/outputs/updater/$(basename "$RUN_DIR")}"
PHYSICAL_GPU="${UPDATER_PHYSICAL_GPU:-}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

telemetry_pid=""
if [[ -n "$PHYSICAL_GPU" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  telemetry_path="$RUN_DIR/gpu_telemetry.csv"
  if [[ ! -s "$telemetry_path" ]]; then
    printf 'timestamp,index,utilization_gpu_percent,memory_used_mib,power_draw_w\n' \
      > "$telemetry_path"
  fi
  (
    while true; do
      nvidia-smi --id="$PHYSICAL_GPU" \
        --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw \
        --format=csv,noheader,nounits >> "$telemetry_path"
      sleep 0.5
    done
  ) &
  telemetry_pid=$!
fi

status=0
"$PYTHON" -m activemap.cli train-updater "$CONFIG" --output "$RUN_DIR" || status=$?
if [[ -n "$telemetry_pid" ]]; then
  kill "$telemetry_pid" 2>/dev/null || true
  wait "$telemetry_pid" 2>/dev/null || true
fi
printf '%s updater job finished with exit_code=%s\n' "$(date --iso-8601=seconds)" "$status"

mkdir -p "$ARCHIVE_DIR/logs"
for log_name in train.log queue.log launch_command.txt; do
  if [[ -f "$RUN_DIR/$log_name" ]]; then
    cp -f "$RUN_DIR/$log_name" "$ARCHIVE_DIR/logs/$log_name"
  fi
done
if [[ -f "$RUN_DIR/train.log" ]]; then
  cp -f "$RUN_DIR/train.log" "$ARCHIVE_DIR/train.log"
fi
printf '%s\n' "$status" > "$ARCHIVE_DIR/logs/exit_code.txt"
exit "$status"
