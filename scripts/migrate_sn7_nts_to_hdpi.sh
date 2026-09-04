#!/usr/bin/env bash
set -euo pipefail

SOURCE_HOST="${SOURCE_HOST:-wh@nts-server.harmo.icu}"
SOURCE_PORT="${SOURCE_PORT:-50053}"
SOURCE_ROOT="${SOURCE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
TARGET_ROOT="${TARGET_ROOT:-/home/wh/ActiveMap}"
TRANSFER_KEY="${TRANSFER_KEY:-/home/wh/.ssh/activemap_transfer_nts_ed25519}"
KEY_COMMENT="${KEY_COMMENT:-activemap-sn7-transfer-20260725}"
SUMMARY="${SUMMARY:-${TARGET_ROOT}/logs/sn7_migration_20260725.summary}"

SSH=(
  ssh
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -i "${TRANSFER_KEY}"
  -p "${SOURCE_PORT}"
)
RSYNC_SHELL="ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ${TRANSFER_KEY} -p ${SOURCE_PORT}"

cleanup_authorization() {
  set +e
  "${SSH[@]}" "${SOURCE_HOST}" \
    "sed -i '/${KEY_COMMENT}\$/d' /home/wh/.ssh/authorized_keys && ! grep -q '${KEY_COMMENT}\$' /home/wh/.ssh/authorized_keys"
  cleanup_status=$?
  if [[ "${cleanup_status}" -eq 0 ]]; then
    rm -f "${TRANSFER_KEY}" "${TRANSFER_KEY}.pub"
    echo "Temporary transfer authorization and key removed."
  else
    echo "WARNING: NTS authorization cleanup failed; retaining HDPI key for manual cleanup." >&2
  fi
  return "${cleanup_status}"
}
trap cleanup_authorization EXIT

test -f "${TRANSFER_KEY}"
mkdir -p \
  "${TARGET_ROOT}/datasets/sn7" \
  "${TARGET_ROOT}/processed/sn7_v1/updater_v4_cap20" \
  "$(dirname "${SUMMARY}")"

rsync -a --partial --human-readable --info=progress2 \
  -e "${RSYNC_SHELL}" \
  "${SOURCE_HOST}:${SOURCE_ROOT}/datasets/sn7/" \
  "${TARGET_ROOT}/datasets/sn7/"

rsync -a --partial --human-readable --info=progress2 \
  -e "${RSYNC_SHELL}" \
  "${SOURCE_HOST}:${SOURCE_ROOT}/processed/sn7_v1/updater_v4_cap20/" \
  "${TARGET_ROOT}/processed/sn7_v1/updater_v4_cap20/"

source_dataset_files="$("${SSH[@]}" "${SOURCE_HOST}" \
  "find '${SOURCE_ROOT}/datasets/sn7' -type f | wc -l")"
target_dataset_files="$(find "${TARGET_ROOT}/datasets/sn7" -type f | wc -l)"
source_processed_files="$("${SSH[@]}" "${SOURCE_HOST}" \
  "find '${SOURCE_ROOT}/processed/sn7_v1/updater_v4_cap20' -type f | wc -l")"
target_processed_files="$(
  find "${TARGET_ROOT}/processed/sn7_v1/updater_v4_cap20" -type f | wc -l
)"
source_manifest_sha="$("${SSH[@]}" "${SOURCE_HOST}" \
  "sha256sum '${SOURCE_ROOT}/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl' | cut -d' ' -f1")"
target_manifest_sha="$(
  sha256sum "${TARGET_ROOT}/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl" |
    cut -d' ' -f1
)"

if [[ "${source_dataset_files}" != "${target_dataset_files}" ]]; then
  echo "Dataset file-count mismatch." >&2
  exit 1
fi
if [[ "${source_processed_files}" != "${target_processed_files}" ]]; then
  echo "Processed file-count mismatch." >&2
  exit 1
fi
if [[ "${source_manifest_sha}" != "${target_manifest_sha}" ]]; then
  echo "Updater manifest hash mismatch." >&2
  exit 1
fi

printf '%s\n' \
  "status=complete" \
  "source_dataset_files=${source_dataset_files}" \
  "target_dataset_files=${target_dataset_files}" \
  "source_processed_files=${source_processed_files}" \
  "target_processed_files=${target_processed_files}" \
  "manifest_sha256=${target_manifest_sha}" \
  "test_assets_used_for_training=false" \
  > "${SUMMARY}"
cat "${SUMMARY}"
