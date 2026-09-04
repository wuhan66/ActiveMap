#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 DOWNLOAD_PID LOG_PATH" >&2
  exit 2
fi

download_pid="$1"
log_path="$2"
data_root="${ACTIVEMAP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap}"
dataset_root="$data_root/datasets/inria_aerial"
archive_root="$dataset_root/archives"
extracted_root="$dataset_root/extracted"
marker="$dataset_root/EXTRACTION_COMPLETE"
declare -A expected=(
  [001]=4294967296
  [002]=4294967296
  [003]=4294967296
  [004]=4294967296
  [005]=3777396691
)

while kill -0 "$download_pid" 2>/dev/null; do
  sleep 120
done

grep -q 'Inria download stage complete' "$log_path"
for part in 001 002 003 004 005; do
  archive="$archive_root/aerialimagelabeling.7z.$part"
  actual="$(stat -c '%s' "$archive")"
  if [[ "$actual" != "${expected[$part]}" ]]; then
    echo "$(basename "$archive") has $actual bytes; expected ${expected[$part]}" >&2
    exit 4
  fi
done

image_count="$(find "$extracted_root" -type f -path '*/train/images/*.tif' | wc -l)"
target_count="$(find "$extracted_root" -type f -path '*/train/gt/*.tif' | wc -l)"
if (( image_count == 0 || image_count != target_count )); then
  echo "incomplete Inria extraction: images=$image_count targets=$target_count" >&2
  exit 5
fi

printf '%s images=%s targets=%s\n' \
  "$(date --iso-8601=seconds)" "$image_count" "$target_count" >"$marker"
