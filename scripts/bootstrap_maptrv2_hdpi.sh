#!/usr/bin/env bash
set -euo pipefail

STORE="${STORE:-/home/wh/ActiveMap}"
CODE_ROOT="${CODE_ROOT:-${STORE}/external/MapTR}"
STATUS_ROOT="${STATUS_ROOT:-${STORE}/datasets/maptrv2}"
SOURCE_URL=https://github.com/hustvl/MapTR.git
SOURCE_COMMIT=a6872d8d9670bde17b4b01560f1221f88b443d55
NUSCENES_ROOT="${NUSCENES_ROOT:-${STORE}/datasets/nuscenes}"

mkdir -p "$(dirname "${CODE_ROOT}")" "${STATUS_ROOT}"
if [[ ! -d "${CODE_ROOT}/.git" ]]; then
  git clone --filter=blob:none "${SOURCE_URL}" "${CODE_ROOT}"
fi
if ! git -C "${CODE_ROOT}" cat-file -e "${SOURCE_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${CODE_ROOT}" fetch origin "${SOURCE_COMMIT}"
fi
git -C "${CODE_ROOT}" checkout --detach "${SOURCE_COMMIT}"
[[ "$(git -C "${CODE_ROOT}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]]

license_file=$(find "${CODE_ROOT}" -maxdepth 2 -type f \( -iname 'license*' -o -iname 'copying*' \) | head -1 || true)
data_state=missing
if [[ -d "${NUSCENES_ROOT}" ]] && find "${NUSCENES_ROOT}" -type f -print -quit | grep -q .; then
  data_state=present_unverified
fi
cat >"${STATUS_ROOT}/BOOTSTRAP_STATUS.json" <<EOF
{"schema_version":"activemap-maptrv2-bootstrap-status-v1","state":"source_locked_data_${data_state}","source_url":"${SOURCE_URL}","source_commit":"${SOURCE_COMMIT}","license_file":"${license_file}","nuscenes_root":"${NUSCENES_ROOT}","test_assets_read":false}
EOF
git -C "${CODE_ROOT}" status --short >"${STATUS_ROOT}/source_status.txt"
git -C "${CODE_ROOT}" log -1 --format=fuller >"${STATUS_ROOT}/source_commit.txt"
cat "${STATUS_ROOT}/BOOTSTRAP_STATUS.json"
