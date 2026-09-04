#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
MODEL_ROOT="${STORAGE_ROOT}/models/opencd_ban"
AUDIT_ROOT="${STORAGE_ROOT}/logs/opencd_ban_pretrained_20260726"
CLIP_NAME="clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
SIDE_NAME="mit_b0_20220624-7e0fe6dd.pth"
CLIP_MIRROR_URL="https://hf-mirror.com/likyoo/BAN/resolve/main/pretrain/${CLIP_NAME}"
CLIP_ORIGIN_URL="https://huggingface.co/likyoo/BAN/resolve/main/pretrain/${CLIP_NAME}"
SIDE_URL="https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/${SIDE_NAME}"

mkdir -p "${MODEL_ROOT}" "${AUDIT_ROOT}"

download_checked() {
  local destination="$1"
  local expected_prefix="$2"
  shift 2
  if [[ ! -s "${destination}" ]]; then
    local downloaded=0
    local url
    for url in "$@"; do
      if curl --fail --location --retry 2 --retry-delay 5 \
        --connect-timeout 15 --max-time 900 \
        --output "${destination}.partial" "${url}"; then
        downloaded=1
        break
      fi
    done
    [[ "${downloaded}" -eq 1 ]]
    mv "${destination}.partial" "${destination}"
  fi
  local digest
  digest="$(sha256sum "${destination}" | cut -d' ' -f1)"
  [[ "${digest}" == "${expected_prefix}"* ]]
  printf '%s  %s\n' "${digest}" "${destination}"
}

download_checked \
  "${MODEL_ROOT}/${CLIP_NAME}" "d08f8887" \
  "${CLIP_MIRROR_URL}" "${CLIP_ORIGIN_URL}" \
  > "${AUDIT_ROOT}/clip_sha256.txt"
download_checked \
  "${MODEL_ROOT}/${SIDE_NAME}" "7e0fe6dd" "${SIDE_URL}" \
  > "${AUDIT_ROOT}/side_sha256.txt"

python3 - "${MODEL_ROOT}" "${AUDIT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

model_root = Path(sys.argv[1])
audit_root = Path(sys.argv[2])
files = sorted(model_root.glob("*.pth"))
payload = {
    "schema_version": "opencd-ban-pretrained-audit-v1",
    "sources": {
        "clip": "https://huggingface.co/likyoo/BAN (hf-mirror transport allowed)",
        "side_encoder": "https://download.openmmlab.com/mmsegmentation/",
    },
    "files": {
        path.name: {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    },
}
(audit_root / "audit.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2))
PY
