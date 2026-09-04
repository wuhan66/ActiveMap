#!/usr/bin/env bash
set -euo pipefail

STORAGE="${STORAGE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE}/envs/activemap-agent/bin/python}"
DATA_ROOT="${DATA_ROOT:-${STORAGE}/datasets/argotweak}"
CODE_ROOT="${CODE_ROOT:-${STORAGE}/external/ArgoTweak_baselines}"
SOURCE_URL="https://github.com/KTH-RPL/ArgoTweak_baselines.git"
SOURCE_COMMIT="016f7cfc3d7d21b533a828d6636a3ae86a1ed415"
SOURCE_LICENSE="CC-BY-NC-SA-4.0"
HF_REPO="lwild/ArgoTweak"

[[ ! -e "${DATA_ROOT}/BOOTSTRAP_COMPLETE.json" ]] || {
  echo "ArgoTweak extension bootstrap already complete"
  exit 0
}
mkdir -p "${DATA_ROOT}/raw/extension_train_val" "${DATA_ROOT}/manifests" "$(dirname "${CODE_ROOT}")"

write_status() {
  local state="$1" detail="$2"
  cat >"${DATA_ROOT}/BOOTSTRAP_STATUS.json" <<EOF
{"schema_version":"activemap-argotweak-bootstrap-status-v1","state":"${state}","detail":"${detail}","source_commit":"${SOURCE_COMMIT}","source_license":"${SOURCE_LICENSE}","hf_repo":"${HF_REPO}","download_scope":"train+val_only","test_assets_read":false}
EOF
}

if [[ ! -d "${CODE_ROOT}/.git" ]]; then
  git clone --filter=blob:none "${SOURCE_URL}" "${CODE_ROOT}"
fi
if ! git -C "${CODE_ROOT}" cat-file -e "${SOURCE_COMMIT}^{commit}" 2>/dev/null; then
  git -C "${CODE_ROOT}" fetch origin "${SOURCE_COMMIT}"
fi
git -C "${CODE_ROOT}" checkout --detach "${SOURCE_COMMIT}"
[[ "$(git -C "${CODE_ROOT}" rev-parse HEAD)" == "${SOURCE_COMMIT}" ]]
write_status "source_locked" "official source is pinned; train+val download pending"

if ! HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  "${PYTHON}" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='${HF_REPO}', repo_type='dataset', local_dir='${DATA_ROOT}/raw/extension_train_val', allow_patterns=['train+val/**', 'README.md'], ignore_patterns=['test/**'])"; then
  write_status "data_download_failed" "Hugging Face train+val download failed; inspect bootstrap.log and retry without reading test"
  exit 4
fi

find "${DATA_ROOT}/raw/extension_train_val" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum >"${DATA_ROOT}/manifests/extension_train_val.sha256"
du -sb "${DATA_ROOT}/raw/extension_train_val" >"${DATA_ROOT}/manifests/extension_train_val.size.txt"
git -C "${CODE_ROOT}" status --short >"${DATA_ROOT}/manifests/source_status.txt"

cat >"${DATA_ROOT}/BOOTSTRAP_COMPLETE.json" <<EOF
{"schema_version":"activemap-argotweak-bootstrap-v1","source_commit":"${SOURCE_COMMIT}","source_license":"${SOURCE_LICENSE}","hf_repo":"${HF_REPO}","download_scope":"train+val_only","test_assets_read":false}
EOF
write_status "complete" "official source and train+val extension are ready"
