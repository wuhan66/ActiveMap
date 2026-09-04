#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "usage: $0 READY_FILE OUTPUT_DIR LOG_PATH NAME=HISTORY_JSONL NAME=HISTORY_JSONL [...]" >&2
  exit 2
fi

ready_file="$1"
output_dir="$2"
log_path="$3"
shift 3
runs=("$@")
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"

mkdir -p "$(dirname "$log_path")" "$output_dir"
exec >>"$log_path" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for $ready_file"
while [[ ! -s "$ready_file" ]]; do
  sleep 120
done

arguments=()
for item in "${runs[@]}"; do
  arguments+=(--run "$item")
done
cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
"$python" scripts/compare_updater_runs.py \
  "${arguments[@]}" \
  --output-dir "$output_dir"
echo "[$(date --iso-8601=seconds)] updater comparison complete"
