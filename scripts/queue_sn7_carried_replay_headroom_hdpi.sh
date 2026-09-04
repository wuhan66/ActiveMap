#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
STATE_FILE="${STATE_FILE:-${STORAGE_ROOT}/runs/updater/sn7_carried_replay_trainval_seed20260902_v2/state.json}"
POLL_SECONDS="${POLL_SECONDS:-180}"

[[ -f "${STATE_FILE}" ]] || {
  echo "missing updater state file: ${STATE_FILE}" >&2
  exit 2
}
[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "POLL_SECONDS must be a positive integer" >&2
  exit 2
}

while true; do
  if grep -q '"status": "completed"' "${STATE_FILE}"; then
    break
  fi
  if grep -q '"status": "failed"' "${STATE_FILE}"; then
    echo "updater training failed; gate will not run" >&2
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

GPU="${GPU:-3}" bash "${PROJECT_ROOT}/scripts/run_sn7_carried_replay_headroom_hdpi.sh"
