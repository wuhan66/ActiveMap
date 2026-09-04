#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
LOG_DIR="${ACTIVEMAP_LOG_ROOT}/dataset_bootstrap"
PID_FILE="${LOG_DIR}/bootstrap.pid"
LOG_FILE="${LOG_DIR}/bootstrap.log"
mkdir -p "${LOG_DIR}"

if [[ -s "${PID_FILE}" ]]; then
  old_pid="$(cat "${PID_FILE}")"
  if kill -0 "${old_pid}" 2>/dev/null; then
    echo "Dataset bootstrap is already running as PID ${old_pid}" >&2
    exit 8
  fi
fi

nohup bash "${PROJECT_ROOT}/scripts/bootstrap_public_datasets.sh" \
  >>"${LOG_FILE}" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "${pid}" >"${PID_FILE}"
echo "Started public dataset bootstrap PID ${pid}; log=${LOG_FILE}"
