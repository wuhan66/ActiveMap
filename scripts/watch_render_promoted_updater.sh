#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "usage: $0 DECISION_JSON SAMPLES_JSONL OUTPUT_PNG LOG_PATH [POLL_SECONDS]" >&2
  exit 2
fi

decision="$1"
samples="$2"
output="$3"
log_path="$4"
poll_seconds="${5:-120}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"

mkdir -p "$(dirname "$log_path")" "$(dirname "$output")"
exec >>"$log_path" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for $decision"
while [[ ! -s "$decision" ]]; do
  sleep "$poll_seconds"
done

cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python" scripts/render_promoted_updater.py \
  "$decision" "$samples" "$output" --split val --device cpu --count 8
"$python" scripts/inspect_updater_checkpoints.py \
  "$(dirname "$decision")" \
  --output "$(dirname "$decision")/checkpoint_inventory_final.json"
echo "[$(date --iso-8601=seconds)] promoted updater rendering complete"
