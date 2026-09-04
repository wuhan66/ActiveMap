#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ActiveMap Python runtime not found: $PYTHON" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if "$PYTHON" -c "import pytest" >/dev/null 2>&1; then
  "$PYTHON" -m pytest \
    "$PROJECT_ROOT/tests/test_sn7_pipeline.py" \
    "$PROJECT_ROOT/tests/test_nn_updater.py" \
    "$PROJECT_ROOT/tests/test_counterfactual_builder.py" \
    -q
else
  "$PYTHON" "$PROJECT_ROOT/scripts/smoke_sn7_v5_temporal_pair.py"
fi
