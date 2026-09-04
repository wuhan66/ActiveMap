#!/usr/bin/env bash
set -euo pipefail

STORAGE="${STORAGE:-/home/wh/ActiveMap}"
CODE_ROOT="${CODE_ROOT:-${STORAGE}/external/MapEx}"
DATA_ROOT="${DATA_ROOT:-${STORAGE}/datasets/mapex_kth}"
SOURCE_URL="https://github.com/castacks/MapEx.git"
SOURCE_COMMIT="53636bd1c79153acc3c74a532837d78c926bae5e"

[[ ! -e "${DATA_ROOT}/BOOTSTRAP_COMPLETE.json" ]] || {
  echo "MapEx KTH bootstrap already complete"
  exit 0
}
mkdir -p "$(dirname "${CODE_ROOT}")" "${DATA_ROOT}/manifests"
if [[ ! -d "${CODE_ROOT}/kth_test_maps" ]]; then
  if [[ ! -d "${CODE_ROOT}/.git" ]]; then
    git clone --filter=blob:none --no-recurse-submodules "${SOURCE_URL}" "${CODE_ROOT}"
  fi
  if ! git -C "${CODE_ROOT}" cat-file -e "${SOURCE_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${CODE_ROOT}" fetch origin "${SOURCE_COMMIT}"
  fi
  git -C "${CODE_ROOT}" checkout --detach "${SOURCE_COMMIT}"
  [[ "$(git -C "${CODE_ROOT}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]]
fi
[[ -d "${CODE_ROOT}/kth_test_maps" ]]

find "${CODE_ROOT}/kth_test_maps" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${DATA_ROOT}/manifests/kth_maps.sha256"
find "${CODE_ROOT}/kth_test_maps" -type f | sort >"${DATA_ROOT}/manifests/kth_maps.files.txt"
du -sb "${CODE_ROOT}/kth_test_maps" >"${DATA_ROOT}/manifests/kth_maps.size.txt"
cat >"${DATA_ROOT}/BOOTSTRAP_COMPLETE.json" <<EOF
{"schema_version":"activemap-mapex-kth-bootstrap-v1","source_commit":"${SOURCE_COMMIT}","asset_scope":"pinned_codeload_kth_maps","evaluation_role":"cross_domain_pilot","test_assets_read":false}
EOF
