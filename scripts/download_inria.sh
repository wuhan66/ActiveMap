#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-/mnt/mydisk/wh/ActiveMap/datasets/inria_aerial}"
EXTRACT="${2:-true}"
BASE_URL="https://files.inria.fr/aerialimagelabeling"
EXTRACTION_COMPLETE="$TARGET_ROOT/EXTRACTION_COMPLETE"
declare -A EXPECTED_BYTES=(
  [001]=4294967296
  [002]=4294967296
  [003]=4294967296
  [004]=4294967296
  [005]=3777396691
)

mkdir -p "$TARGET_ROOT/archives"
for part in 001 002 003 004 005; do
  filename="aerialimagelabeling.7z.$part"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c --continue=true --allow-overwrite=true --auto-file-renaming=false \
      --check-certificate=false --file-allocation=none \
      --max-connection-per-server=16 --split=16 --min-split-size=4M \
      --summary-interval=30 --dir "$TARGET_ROOT/archives" --out "$filename" \
      "$BASE_URL/$filename"
  else
    wget -c --no-check-certificate --progress=dot:giga \
      "$BASE_URL/$filename" -O "$TARGET_ROOT/archives/$filename"
  fi
  actual_bytes="$(stat -c '%s' "$TARGET_ROOT/archives/$filename")"
  if [[ "$actual_bytes" != "${EXPECTED_BYTES[$part]}" ]]; then
    echo "$filename has $actual_bytes bytes; expected ${EXPECTED_BYTES[$part]}" >&2
    exit 4
  fi
done
if [[ "$EXTRACT" == "true" ]]; then
  rm -f "$EXTRACTION_COMPLETE"
  command -v 7z >/dev/null 2>&1 || {
    echo "7z is unavailable; archives are complete but extraction is pending" >&2
    exit 2
  }
  mkdir -p "$TARGET_ROOT/extracted"
  7z t "$TARGET_ROOT/archives/aerialimagelabeling.7z.001"
  7z x -y "$TARGET_ROOT/archives/aerialimagelabeling.7z.001" \
    -o"$TARGET_ROOT/extracted"
  nested="$TARGET_ROOT/extracted/NEW2-AerialImageDataset.zip"
  if [[ -f "$nested" ]]; then
    command -v unzip >/dev/null 2>&1 || {
      echo "unzip is required for the nested Inria archive" >&2
      exit 3
    }
    unzip -q -n "$nested" -d "$TARGET_ROOT/extracted"
  fi
  printf '%s\n' "$(date --iso-8601=seconds)" >"$EXTRACTION_COMPLETE"
fi
printf '%s Inria download stage complete\n' "$(date --iso-8601=seconds)"
