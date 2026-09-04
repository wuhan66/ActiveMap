#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
DATA_ROOT="${DATA_ROOT:-${STORAGE_ROOT}/datasets/spacenet8}"
ARCHIVE="Germany_Training_Public.tar.gz"
EXPECTED_BYTES=1357293885
S3_URI="s3://spacenet-dataset/spacenet/SN8_floods/tarballs/${ARCHIVE}"
AWS_BIN="${AWS_BIN:-/snap/bin/aws}"

mkdir -p "${DATA_ROOT}/raw" "${DATA_ROOT}/manifests"
target="${DATA_ROOT}/raw/${ARCHIVE}"
if [[ ! -f "${target}" ]]; then
  "${AWS_BIN}" s3 cp --no-sign-request "${S3_URI}" "${target}.partial"
  mv "${target}.partial" "${target}"
fi
actual_bytes="$(stat -c %s "${target}")"
[[ "${actual_bytes}" -eq "${EXPECTED_BYTES}" ]] || {
  echo "Unexpected archive size: ${actual_bytes}" >&2
  exit 3
}
sha256sum "${target}" >"${DATA_ROOT}/manifests/${ARCHIVE}.sha256"
tar -tzf "${target}" | sort >"${DATA_ROOT}/manifests/${ARCHIVE}.files.txt"
cat >"${DATA_ROOT}/BOOTSTRAP_COMPLETE.json" <<EOF
{"schema_version":"activemap-spacenet8-bootstrap-v1","scope":"germany_training_public","archive_bytes":${actual_bytes},"license":"CC-BY-SA-4.0","test_assets_read":false}
EOF
