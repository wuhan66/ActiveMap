#!/usr/bin/env bash
set -euo pipefail

output_root="${1:-/mnt/mydisk/wh/ActiveMap/models/sam_road}"
mkdir -p "$output_root"
sam_road_url="${SAM_ROAD_MODEL_URL:-https://hf-mirror.com/congrui/sam_road/resolve/main/spacenet_vitb_256_e10.ckpt}"
sam_vit_b_url="${SAM_VIT_B_URL:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth}"

download_once() {
  local url="$1"
  local target="$2"
  if [[ -e "$target" ]]; then
    echo "Refusing to overwrite existing weight: $target" >&2
    return 2
  fi
  if command -v curl >/dev/null 2>&1; then
    curl --fail --location --continue-at - --output "$target.part" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget --continue --output-document "$target.part" "$url"
  else
    echo "Neither curl nor wget is available" >&2
    return 127
  fi
  mv "$target.part" "$target"
  sha256sum "$target"
}

download_once \
  "$sam_road_url" \
  "$output_root/spacenet_vitb_256_e10.ckpt"
download_once \
  "$sam_vit_b_url" \
  "$output_root/sam_vit_b_01ec64.pth"
