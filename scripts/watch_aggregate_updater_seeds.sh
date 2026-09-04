#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 LOG_PATH OUTPUT_JSON SEED=DECISION_JSON SEED=DECISION_JSON [...]" >&2
  exit 2
fi

log_path="$1"
output_json="$2"
shift 2
decisions=("$@")

mkdir -p "$(dirname "$log_path")" "$(dirname "$output_json")"
exec >>"$log_path" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for ${#decisions[@]} seed decisions"

for item in "${decisions[@]}"; do
  decision_path="${item#*=}"
  while [[ ! -s "$decision_path" ]]; do
    sleep 120
  done
done

arguments=()
for item in "${decisions[@]}"; do
  arguments+=(--decision "$item")
done
export PYTHONPATH="${PYTHONPATH:-}:src"
/home/wh/venvs/activemap/bin/python scripts/aggregate_updater_seeds.py \
  "${arguments[@]}" \
  --output "$output_json"
echo "[$(date --iso-8601=seconds)] seed aggregation complete"
