#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
DATA_ROOT="${ACTIVEMAP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

MUNO_OUTPUT="$DATA_ROOT/processed/muno21_v1/updater"
INRIA_OUTPUT="$DATA_ROOT/processed/inria_v1/updater"
SN7_MANIFEST="$DATA_ROOT/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl"
MULTIDATA_OUTPUT="$DATA_ROOT/processed/multidata_v1"

bash scripts/prepare_muno21_updater.sh

bash scripts/prepare_inria_updater.sh

mkdir -p "$MULTIDATA_OUTPUT"
"$PYTHON" -m activemap.cli merge-updater-manifests \
  "$MULTIDATA_OUTPUT/updater_samples.jsonl" \
  "$SN7_MANIFEST" \
  "$MUNO_OUTPUT/updater_samples.jsonl" \
  "$INRIA_OUTPUT/updater_samples.jsonl"
