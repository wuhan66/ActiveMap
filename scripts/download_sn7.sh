#!/usr/bin/env bash
set -euo pipefail

DESTINATION="${1:-data/raw/sn7}"
EXTRACT="${2:-false}"
BUCKET="spacenet-dataset"
KEY="spacenet/SN7_buildings/tarballs/SN7_buildings_train.tar.gz"
PUBLIC_URL="https://spacenet-dataset.s3.amazonaws.com/$KEY"
ARCHIVE_NAME="SN7_buildings_train.tar.gz"
MINIMUM_FREE_BYTES=$((30 * 1024 * 1024 * 1024))

if command -v aws >/dev/null 2>&1; then
  DOWNLOAD_BACKEND="aws"
elif command -v aria2c >/dev/null 2>&1; then
  DOWNLOAD_BACKEND="aria2"
elif command -v wget >/dev/null 2>&1; then
  DOWNLOAD_BACKEND="wget"
else
  echo "Either AWS CLI v2 or wget is required."
  exit 1
fi

mkdir -p "$DESTINATION"
AVAILABLE_BYTES="$(df --output=avail -B1 "$DESTINATION" | tail -n 1 | tr -d ' ')"
if (( AVAILABLE_BYTES < MINIMUM_FREE_BYTES )); then
  echo "At least 30 GiB free space is required in $DESTINATION."
  exit 1
fi

if [[ "$DOWNLOAD_BACKEND" == "aws" ]]; then
  EXPECTED_BYTES="$(aws s3api head-object --bucket "$BUCKET" --key "$KEY" --no-sign-request --query ContentLength --output text)"
else
  EXPECTED_BYTES="$(wget --spider --server-response "$PUBLIC_URL" 2>&1 | awk 'tolower($1) == "content-length:" {print $2}' | tail -n 1 | tr -d '\r')"
fi
if [[ -z "$EXPECTED_BYTES" || ! "$EXPECTED_BYTES" =~ ^[0-9]+$ ]]; then
  echo "Unable to determine the remote SpaceNet 7 archive size."
  exit 1
fi
ARCHIVE_PATH="$DESTINATION/$ARCHIVE_NAME"
CURRENT_BYTES=0
[[ -f "$ARCHIVE_PATH" ]] && CURRENT_BYTES="$(stat -c %s "$ARCHIVE_PATH")"
if [[ "$CURRENT_BYTES" != "$EXPECTED_BYTES" ]]; then
  if [[ "$DOWNLOAD_BACKEND" == "aws" ]]; then
    aws s3 cp "s3://$BUCKET/$KEY" "$ARCHIVE_PATH" --no-sign-request --only-show-errors
  elif [[ "$DOWNLOAD_BACKEND" == "aria2" ]]; then
    aria2c --continue=true --allow-overwrite=true --auto-file-renaming=false \
      --file-allocation=none --max-connection-per-server=16 --split=16 \
      --min-split-size=4M --summary-interval=30 \
      --dir "$DESTINATION" --out "$ARCHIVE_NAME" "$PUBLIC_URL"
  else
    wget --continue --output-document="$ARCHIVE_PATH" "$PUBLIC_URL"
  fi
fi

DOWNLOADED_BYTES="$(stat -c %s "$ARCHIVE_PATH")"
[[ "$DOWNLOADED_BYTES" == "$EXPECTED_BYTES" ]] || {
  echo "Archive size mismatch: expected $EXPECTED_BYTES, found $DOWNLOADED_BYTES."
  exit 1
}
sha256sum "$ARCHIVE_PATH"

if [[ "$EXTRACT" == "true" ]]; then
  mkdir -p "$DESTINATION/train"
  tar -xzf "$ARCHIVE_PATH" -C "$DESTINATION/train"
fi
