#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 DATA_ROOT RUN_DIR PHYSICAL_GPU_INDEX [POLL_SECONDS]" >&2
  exit 2
fi

DATA_ROOT="$1"
RUN_DIR="$2"
GPU_INDEX="$3"
POLL_SECONDS="${4:-30}"
if ! [[ "$GPU_INDEX" =~ ^[0-9]+$ && "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPU index must be non-negative and poll seconds must be positive" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUDIT="$DATA_ROOT/audit.json"
TRAIN="$DATA_ROOT/train.jsonl"
VAL="$DATA_ROOT/val.jsonl"

echo "waiting_for=$AUDIT gpu=$GPU_INDEX poll_seconds=$POLL_SECONDS"
while [[ ! -s "$AUDIT" ]]; do
  sleep "$POLL_SECONDS"
done

/home/wh/venvs/activemap/bin/python - "$AUDIT" <<'PY'
import json
import sys

path = sys.argv[1]
report = json.load(open(path, encoding="utf-8"))
if report.get("passed") is not True:
    raise SystemExit(f"audit did not pass: {path}")
PY

if pgrep -u "$USER" -f '[p]ython .*scripts/train_' >/dev/null; then
  echo "refusing_start=another ActiveMap trainer is active" >&2
  exit 3
fi

GPU_UUID="$({ nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits || true; } \
  | awk -F', *' -v gpu_index="$GPU_INDEX" '$1 == gpu_index {print $2}')"
if [[ -z "$GPU_UUID" ]]; then
  echo "refusing_start=GPU index $GPU_INDEX does not exist" >&2
  exit 3
fi
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
  | grep -Fxq "$GPU_UUID"; then
  echo "refusing_start=GPU $GPU_INDEX is occupied" >&2
  exit 3
fi

echo "training_start=$(date --iso-8601=seconds) gpu=$GPU_INDEX run=$RUN_DIR"
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_INDEX" PYTHONPATH=src \
  /home/wh/venvs/activemap/bin/python scripts/train_tool_belief.py \
  "$TRAIN" "$VAL" "$RUN_DIR" \
  --audit-report "$AUDIT" \
  --device cuda:0 \
  --seed 20260821
echo "training_complete=$(date --iso-8601=seconds) run=$RUN_DIR"
