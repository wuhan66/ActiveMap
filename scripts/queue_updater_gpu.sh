#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 5 ]]; then
  echo "usage: $0 GPU_INDEX CONFIG LOG_PATH [MAX_USED_MIB] [POLL_SECONDS]" >&2
  exit 2
fi

gpu_index="$1"
config="$2"
log_path="$3"
max_used_mib="${4:-1024}"
poll_seconds="${5:-30}"
required_idle_checks=3

if [[ ! -f "$config" ]]; then
  echo "config not found: $config" >&2
  exit 2
fi
if ! [[ "$gpu_index" =~ ^[0-9]+$ && "$max_used_mib" =~ ^[0-9]+$ && "$poll_seconds" =~ ^[0-9]+$ ]]; then
  echo "GPU_INDEX, MAX_USED_MIB, and POLL_SECONDS must be non-negative integers" >&2
  exit 2
fi

mkdir -p "$(dirname "$log_path")"
exec >>"$log_path" 2>&1

echo "[$(date --iso-8601=seconds)] queued config=$config gpu=$gpu_index max_used_mib=$max_used_mib"
idle_checks=0
while (( idle_checks < required_idle_checks )); do
  used_mib="$(
    nvidia-smi --id="$gpu_index" --query-gpu=memory.used --format=csv,noheader,nounits |
      tr -d '[:space:]'
  )"
  if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib <= max_used_mib )); then
    ((idle_checks += 1)) || true
    echo "[$(date --iso-8601=seconds)] idle_check=$idle_checks/$required_idle_checks used_mib=$used_mib"
  else
    idle_checks=0
  fi
  if (( idle_checks < required_idle_checks )); then
    sleep "$poll_seconds"
  fi
done

echo "[$(date --iso-8601=seconds)] starting config=$config on gpu=$gpu_index"
export CUDA_VISIBLE_DEVICES="$gpu_index"
export PYTHONPATH="${PYTHONPATH:-}:src"
exec /home/wh/venvs/activemap/bin/python -m activemap.cli train-updater "$config"
