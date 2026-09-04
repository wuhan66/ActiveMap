#!/usr/bin/env bash
set -euo pipefail

TARGET_ROOT="${1:-/mnt/mydisk/wh/ActiveMap/datasets/muno21}"
EXTRACT="${2:-true}"
URL="https://favyen.com/files/muno21.zip"
ARCHIVE="$TARGET_ROOT/muno21.zip"

mkdir -p "$TARGET_ROOT"
printf '%s downloading MUNO21 to %s\n' "$(date --iso-8601=seconds)" "$ARCHIVE"
if command -v aria2c >/dev/null 2>&1; then
  aria2c --continue=true --allow-overwrite=true --auto-file-renaming=false \
    --file-allocation=none --max-connection-per-server=16 --split=16 \
    --min-split-size=4M --summary-interval=30 \
    --dir "$TARGET_ROOT" --out "$(basename "$ARCHIVE")" "$URL"
else
  wget -c --progress=dot:giga "$URL" -O "$ARCHIVE"
fi
if [[ "$EXTRACT" == "true" ]]; then
  command -v unzip >/dev/null 2>&1 || {
    echo "unzip is required to extract MUNO21" >&2
    exit 2
  }
  mkdir -p "$TARGET_ROOT/extracted"
  unzip -q -n "$ARCHIVE" -d "$TARGET_ROOT/extracted"
fi
printf '%s MUNO21 download stage complete\n' "$(date --iso-8601=seconds)"
