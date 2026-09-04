#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ACTION="${1:-status}"
CONFIG="${ACTIVEMAP_DEPLOY_CONFIG:-$PROJECT_ROOT/configs/deployment/server.yaml}"
PYTHON="${ACTIVEMAP_SERVE_PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_DIR="${ACTIVEMAP_SERVE_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/model_server}"
PID_FILE="$RUN_DIR/server.pid"
LOG_FILE="$RUN_DIR/server.log"
mkdir -p "$RUN_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null
}

case "$ACTION" in
  start)
    if is_running; then
      echo "ActiveMap model server already running: PID $(cat "$PID_FILE")"
      exit 0
    fi
    "$PYTHON" scripts/model_doctor.py "$CONFIG" --require agent_qwen3_4b
    nohup "$PYTHON" scripts/serve_models.py "$CONFIG" >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 3
    if ! is_running; then
      echo "Model server failed to start; inspect $LOG_FILE" >&2
      exit 1
    fi
    echo "ActiveMap model server started: PID $(cat "$PID_FILE")"
    ;;
  stop)
    if is_running; then
      kill "$(cat "$PID_FILE")"
      for _ in {1..20}; do
        is_running || break
        sleep 1
      done
      is_running && kill -KILL "$(cat "$PID_FILE")"
    fi
    rm -f "$PID_FILE"
    echo "ActiveMap model server stopped"
    ;;
  restart)
    "$0" stop
    "$0" start
    ;;
  status)
    if is_running; then
      echo "running PID $(cat "$PID_FILE")"
      "$PYTHON" -c "import json,urllib.request; print(json.dumps(json.load(urllib.request.urlopen('http://127.0.0.1:8008/health')), indent=2))"
    else
      echo "stopped"
      "$PYTHON" scripts/model_doctor.py "$CONFIG"
    fi
    ;;
  logs)
    tail -n 100 "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 start|stop|restart|status|logs" >&2
    exit 2
    ;;
esac
