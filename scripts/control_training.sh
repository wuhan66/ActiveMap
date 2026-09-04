#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 RUN_ROOT pause|resume|stop|clear-stop|status" >&2
  exit 2
fi

RUN_ROOT="$1"
ACTION="$2"
CONTROL_DIR="$RUN_ROOT/control"
mkdir -p "$CONTROL_DIR"

apply_to_runs() {
  local marker="$1"
  local operation="$2"
  for directory in "$RUN_ROOT/updater/control" "$RUN_ROOT/selector/control"; do
    mkdir -p "$directory"
    if [[ "$operation" == "touch" ]]; then
      touch "$directory/$marker"
    else
      rm -f "$directory/$marker"
    fi
  done
}

case "$ACTION" in
  pause)
    touch "$CONTROL_DIR/PAUSE"
    apply_to_runs PAUSE touch
    ;;
  resume)
    rm -f "$CONTROL_DIR/PAUSE"
    apply_to_runs PAUSE remove
    ;;
  stop)
    touch "$CONTROL_DIR/STOP"
    apply_to_runs STOP touch
    ;;
  clear-stop)
    rm -f "$CONTROL_DIR/STOP"
    apply_to_runs STOP remove
    ;;
  status)
    [[ -f "$RUN_ROOT/nightly_state.json" ]] && cat "$RUN_ROOT/nightly_state.json"
    for state in "$RUN_ROOT"/*/state.json; do
      [[ -f "$state" ]] && printf '\n%s\n' "$(cat "$state")"
    done
    ;;
  *)
    echo "Unknown action: $ACTION" >&2
    exit 2
    ;;
esac
