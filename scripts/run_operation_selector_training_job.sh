#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${OPERATION_SELECTOR_RUN_DIR:?OPERATION_SELECTOR_RUN_DIR is required}"
CONFIG="${OPERATION_SELECTOR_CONFIG:?OPERATION_SELECTOR_CONFIG is required}"
PYTHON="${ACTIVEMAP_PYTHON:?ACTIVEMAP_PYTHON is required}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

status=0
"$PYTHON" -m activemap.cli train-operation-selector "$CONFIG" --output "$RUN_DIR" || status=$?
printf '%s operation selector job finished with exit_code=%s\n' "$(date --iso-8601=seconds)" "$status"
printf '%s\n' "$status" > "$RUN_DIR/exit_code.txt"
exit "$status"
