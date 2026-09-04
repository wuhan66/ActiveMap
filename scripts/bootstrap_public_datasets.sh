#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"

LOG_DIR="${ACTIVEMAP_LOG_ROOT}/dataset_bootstrap"
ARTIFACT_DIR="${ACTIVEMAP_STORAGE_ROOT}/artifacts/datasets"
STATUS_FILE="${LOG_DIR}/status.tsv"
LOCK_FILE="${LOG_DIR}/bootstrap.lock"
mkdir -p "${ACTIVEMAP_DATASET_ROOT}" "${LOG_DIR}" "${ARTIFACT_DIR}"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another dataset bootstrap process holds ${LOCK_FILE}" >&2
  exit 9
fi

record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "${STATUS_FILE}"
}

record_hashes() {
  local output="$1"
  shift
  sha256sum "$@" >"${output}.partial"
  mv "${output}.partial" "${output}"
}

cd "${PROJECT_ROOT}"

record_status muno21 downloading
bash scripts/download_muno21.sh "${ACTIVEMAP_DATASET_ROOT}/muno21" true
record_hashes "${ARTIFACT_DIR}/muno21.sha256" \
  "${ACTIVEMAP_DATASET_ROOT}/muno21/muno21.zip"
record_status muno21 ready

record_status sn7 downloading
bash scripts/download_sn7.sh "${ACTIVEMAP_DATASET_ROOT}/sn7" true
record_hashes "${ARTIFACT_DIR}/sn7.sha256" \
  "${ACTIVEMAP_DATASET_ROOT}/sn7/SN7_buildings_train.tar.gz"
record_status sn7 ready

record_status inria downloading
bash scripts/download_inria.sh "${ACTIVEMAP_DATASET_ROOT}/inria_aerial" true
record_hashes "${ARTIFACT_DIR}/inria.sha256" \
  "${ACTIVEMAP_DATASET_ROOT}/inria_aerial/archives/"aerialimagelabeling.7z.*
record_status inria ready

record_status all complete
