#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 7 ]]; then
  echo "usage: $0 READY_FILE GPU_INDEX CONFIG LOG_PATH MAX_USED_MIB [POLL_SECONDS] [READY_POLL_SECONDS]" >&2
  exit 2
fi

ready_file="$1"
gpu_index="$2"
config="$3"
log_path="$4"
max_used_mib="$5"
poll_seconds="${6:-30}"
ready_poll_seconds="${7:-120}"

while [[ ! -s "$ready_file" ]]; do
  sleep "$ready_poll_seconds"
done

exec bash scripts/queue_updater_gpu.sh \
  "$gpu_index" "$config" "$log_path" "$max_used_mib" "$poll_seconds"
