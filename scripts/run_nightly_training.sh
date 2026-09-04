#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
RUN_ROOT="${RUN_ROOT:-/mnt/mydisk/wh/ActiveMap/runs/nightly/$(date +%Y%m%d_%H%M%S)}"
STAGES="${STAGES:-updater}"
DRY_RUN="${DRY_RUN:-0}"
ACTIVEMAP_PYTHON="${ACTIVEMAP_PYTHON:-python}"
UPDATER_CONFIG="${UPDATER_CONFIG:-$PROJECT_ROOT/configs/updater/sn7_v3_server.yaml}"
UPDATER_SAMPLES="${UPDATER_SAMPLES:-/mnt/mydisk/wh/ActiveMap/processed/sn7_v1/updater_v3/updater_samples.jsonl}"
RSP_DATA_ROOT="${RSP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap/processed/sn7_v1/rsprompter_v1}"
RSP_REPOSITORY="${RSP_REPOSITORY:-$PROJECT_ROOT/third_party/RSPrompter}"
RSP_PYTHON="${RSP_PYTHON:-/home/wh/miniconda3/envs/rsprompter/bin/python}"
RSP_CONFIG="${RSP_CONFIG:-}"
RSP_WORK_DIR="${RSP_WORK_DIR:-$RUN_ROOT/rsprompter}"
RSP_RESUME="${RSP_RESUME:-1}"

CONTROL_DIR="$RUN_ROOT/control"
LOG_DIR="$RUN_ROOT/logs"
STATE_FILE="$RUN_ROOT/nightly_state.json"
mkdir -p "$CONTROL_DIR" "$LOG_DIR"

write_state() {
  local status="$1"
  local stage="${2:-none}"
  printf '{"status":"%s","stage":"%s","pid":%s,"updated_at":"%s"}\n' \
    "$status" "$stage" "$$" "$(date --iso-8601=seconds)" > "$STATE_FILE.tmp"
  mv "$STATE_FILE.tmp" "$STATE_FILE"
}

wait_for_control() {
  local stage="$1"
  while [[ -f "$CONTROL_DIR/PAUSE" && ! -f "$CONTROL_DIR/STOP" ]]; do
    write_state "paused" "$stage"
    sleep 10
  done
  [[ ! -f "$CONTROL_DIR/STOP" ]]
}

run_stage() {
  local stage="$1"
  shift
  if [[ -f "$RUN_ROOT/$stage.done" ]]; then
    echo "[$(date --iso-8601=seconds)] skip completed stage: $stage"
    return 0
  fi
  wait_for_control "$stage" || return 20
  write_state "running" "$stage"
  printf '%q ' "$@" > "$LOG_DIR/$stage.command.txt"
  printf '\n' >> "$LOG_DIR/$stage.command.txt"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN [$stage]: $(cat "$LOG_DIR/$stage.command.txt")"
    return 0
  fi
  touch "$RUN_ROOT/$stage.running"
  if "$@" >> "$LOG_DIR/$stage.log" 2>&1; then
    rm -f "$RUN_ROOT/$stage.running"
    touch "$RUN_ROOT/$stage.done"
  else
    local code=$?
    rm -f "$RUN_ROOT/$stage.running"
    printf '%s\n' "$code" > "$RUN_ROOT/$stage.failed"
    write_state "failed" "$stage"
    return "$code"
  fi
}

on_exit() {
  local code=$?
  if [[ "$code" -ne 0 && ! -f "$CONTROL_DIR/STOP" ]]; then
    write_state "failed" "${CURRENT_STAGE:-startup}"
  fi
}
trap on_exit EXIT
trap 'touch "$CONTROL_DIR/STOP"' INT TERM

if [[ ! -f "$UPDATER_SAMPLES" ]]; then
  echo "Missing updater samples: $UPDATER_SAMPLES" >&2
  exit 2
fi
if ! command -v "$ACTIVEMAP_PYTHON" >/dev/null 2>&1 && [[ ! -x "$ACTIVEMAP_PYTHON" ]]; then
  echo "ActiveMap Python is not executable: $ACTIVEMAP_PYTHON" >&2
  exit 2
fi

write_state "starting" "preflight"
for CURRENT_STAGE in $STAGES; do
  case "$CURRENT_STAGE" in
    updater)
      run_stage updater "$ACTIVEMAP_PYTHON" -m activemap.cli train-updater \
        "$UPDATER_CONFIG" --output "$RUN_ROOT/updater" || break
      ;;
    rsp_export)
      run_stage rsp_export "$ACTIVEMAP_PYTHON" -m activemap.cli export-rsprompter-data \
        "$UPDATER_SAMPLES" "$RSP_DATA_ROOT" || break
      ;;
    rsp_train)
      if [[ ! -f "$RSP_REPOSITORY/tools/train.py" ]]; then
        echo "RSPrompter repository is not ready: $RSP_REPOSITORY" >&2
        exit 3
      fi
      if [[ ! -x "$RSP_PYTHON" || -z "$RSP_CONFIG" || ! -f "$RSP_CONFIG" ]]; then
        echo "Set a valid RSP_PYTHON and RSP_CONFIG before enabling rsp_train" >&2
        exit 3
      fi
      rsp_args=(
        "$RSP_PYTHON" "$RSP_REPOSITORY/tools/train.py" "$RSP_CONFIG"
        --cfg-options "data_root=$RSP_DATA_ROOT" "work_dir=$RSP_WORK_DIR"
      )
      if [[ "$RSP_RESUME" == "1" ]]; then
        rsp_args+=(resume=True)
      fi
      run_stage rsp_train "${rsp_args[@]}" || break
      ;;
    *)
      echo "Unknown stage: $CURRENT_STAGE" >&2
      exit 2
      ;;
  esac
done

if [[ -f "$CONTROL_DIR/STOP" ]]; then
  write_state "stopped" "${CURRENT_STAGE:-none}"
  exit 20
fi
write_state "completed" "all"
