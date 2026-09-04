#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 GPU_INDEX RUN_DIR SAMPLES LOG_PATH [FINALIZER_ARGS...]" >&2
  exit 2
fi

gpu_index="$1"
run_dir="$2"
samples="$3"
log_path="$4"
shift 4
state_path="$run_dir/state.json"

mkdir -p "$(dirname "$log_path")"
exec >>"$log_path" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for $state_path"

while ! grep -q '"status": "completed"' "$state_path" 2>/dev/null; do
  sleep 120
done

echo "[$(date --iso-8601=seconds)] training complete; starting validation gate"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONPATH="${PYTHONPATH:-}:src"
/home/wh/venvs/activemap/bin/python scripts/finalize_updater_hierarchy.py \
  "$run_dir" \
  "$samples" \
  --device auto \
  --batch-size 64 \
  --max-false-edit 0.05 \
  --grid-steps 33 \
  --baseline-macro-f1 0.789307 \
  --baseline-delete-f1 0.365979 \
  --bootstrap 1000 \
  "$@"
echo "[$(date --iso-8601=seconds)] finalization complete"
