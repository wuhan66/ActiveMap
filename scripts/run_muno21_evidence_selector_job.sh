#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${MUNO21_SELECTOR_RUN_DIR:?MUNO21_SELECTOR_RUN_DIR is required}"
CONFIG="${MUNO21_SELECTOR_CONFIG:?MUNO21_SELECTOR_CONFIG is required}"
PYTHON="${ACTIVEMAP_PYTHON:?ACTIVEMAP_PYTHON is required}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
status=0
"$PYTHON" -m activemap.cli train-selector "$CONFIG" --output "$RUN_DIR" || status=$?
printf '%s evidence selector job finished with exit_code=%s\n' \
  "$(date --iso-8601=seconds)" "$status"
printf '%s\n' "$status" > "$RUN_DIR/exit_code.txt"
exit "$status"
