#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 LOG_PATH [POLL_SECONDS]" >&2
  exit 2
fi

log_path="$1"
poll_seconds="${2:-300}"
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${ACTIVEMAP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap}"
inria_root="$data_root/datasets/inria_aerial/extracted"
extraction_complete="$data_root/datasets/inria_aerial/EXTRACTION_COMPLETE"

mkdir -p "$(dirname "$log_path")"
exec >>"$log_path" 2>&1
echo "[$(date --iso-8601=seconds)] waiting for completed Inria extraction and train pairs"
while true; do
  image="$(find "$inria_root" -type f -path '*/train/images/*.tif' -print -quit 2>/dev/null || true)"
  target="$(find "$inria_root" -type f -path '*/train/gt/*.tif' -print -quit 2>/dev/null || true)"
  if [[ -s "$extraction_complete" && -n "$image" && -n "$target" ]]; then
    break
  fi
  sleep "$poll_seconds"
done

cd "$project_root"
exec bash scripts/prepare_inria_updater.sh
