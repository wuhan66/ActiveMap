#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 9 || $# -gt 11 ]]; then
  echo "usage: $0 READY_FILE GPU_INDEX CHECKPOINT SAMPLES OUTPUT_DIR PRESENCE_THRESHOLD CHANGE_THRESHOLD LOG_PATH MAX_USED_MIB [POLL_SECONDS] [READY_POLL_SECONDS]" >&2
  exit 2
fi

ready_file="$1"
gpu_index="$2"
checkpoint="$3"
samples="$4"
output_dir="$5"
presence_threshold="$6"
change_threshold="$7"
log_path="$8"
max_used_mib="$9"
poll_seconds="${10:-30}"
ready_poll_seconds="${11:-120}"

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"

mkdir -p "$(dirname "$log_path")" "$output_dir"
exec >>"$log_path" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for $ready_file"
while [[ ! -s "$ready_file" ]]; do
  sleep "$ready_poll_seconds"
done

while true; do
  used_mib="$(nvidia-smi -i "$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
  if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib <= max_used_mib )); then
    break
  fi
  echo "[$(date --iso-8601=seconds)] GPU $gpu_index uses ${used_mib} MiB; waiting"
  sleep "$poll_seconds"
done

cd "$project_root"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[$(date --iso-8601=seconds)] starting validation evaluation on GPU $gpu_index"
CUDA_VISIBLE_DEVICES="$gpu_index" "$python" -m activemap.cli evaluate-updater \
  "$checkpoint" \
  "$samples" \
  "$output_dir" \
  --split val \
  --device cuda \
  --batch-size 64 \
  --num-workers 4 \
  --bootstrap 1000 \
  --presence-threshold "$presence_threshold" \
  --change-threshold "$change_threshold"
echo "[$(date --iso-8601=seconds)] validation evaluation complete"
