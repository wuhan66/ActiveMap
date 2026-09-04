#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
CONFIG="${SN7_V5_CONFIG:-$PROJECT_ROOT/configs/updater/sn7_v5_temporal_pair_seed20260816_server.yaml}"
SAMPLES="${SN7_V5_SAMPLES:-/home/wh/ActiveMap/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1/updater_samples.jsonl}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ActiveMap Python runtime not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$CONFIG" || ! -f "$SAMPLES" ]]; then
  echo "V5 config or paired-temporal samples are missing" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m activemap.cli train-updater "$CONFIG"
