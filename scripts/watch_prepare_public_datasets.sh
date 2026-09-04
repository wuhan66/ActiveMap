#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
DOWNLOAD_STATUS="${ACTIVEMAP_LOG_ROOT}/dataset_bootstrap/status.tsv"
DOWNLOAD_PID="${ACTIVEMAP_LOG_ROOT}/dataset_bootstrap/bootstrap.pid"
LOG_DIR="${ACTIVEMAP_LOG_ROOT}/dataset_preparation"
PID_FILE="${LOG_DIR}/watcher.pid"
LOG_FILE="${LOG_DIR}/preparation.log"
mkdir -p "${LOG_DIR}"
printf '%s\n' "$$" >"${PID_FILE}"

while ! grep -q $'^.*\tall\tcomplete$' "${DOWNLOAD_STATUS}" 2>/dev/null; do
  if [[ -s "${DOWNLOAD_PID}" ]]; then
    download_pid="$(cat "${DOWNLOAD_PID}")"
    if ! kill -0 "${download_pid}" 2>/dev/null; then
      echo "Dataset bootstrap exited before recording all/complete" >&2
      exit 7
    fi
  fi
  sleep 60
done

bash "${PROJECT_ROOT}/scripts/prepare_public_datasets_hdpi.sh" \
  >>"${LOG_FILE}" 2>&1
