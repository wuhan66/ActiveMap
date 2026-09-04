#!/usr/bin/env bash
set -euo pipefail

# Wait for the C5 state-cache audit, then run the one-seed updater-conditioned
# selector pilot on two selected HDPI GPUs.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_updater_conditioned_matrix_v3}"
QUEUE_ROOT="${RUN_ROOT}/selector_pilot_queue_v1"
F0_GPU="${F0_GPU:-3}"
F1_GPU="${F1_GPU:-5}"
POLL_SECONDS="${POLL_SECONDS:-180}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-43200}"

mkdir -p "${QUEUE_ROOT}"
exec 9>"${QUEUE_ROOT}/.queue.lock"
flock -n 9 || exit 0
[[ ! -e "${RUN_ROOT}/SELECTOR_PILOT_COMPLETE.json" ]] || exit 0
[[ ! -e "${QUEUE_ROOT}/FAILED.json" ]] || exit 3

deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
printf '%s waiting for C5 state audit at %s\n' "$(date -Is)" "${RUN_ROOT}" >"${QUEUE_ROOT}/queue.log"
while [[ ! -s "${RUN_ROOT}/STATE_AUDIT_COMPLETE.json" ]]; do
  if [[ -s "${RUN_ROOT}/STATE_AUDIT_FAILED.json" ]]; then
    printf '{"status":"failed","reason":"state_audit_failed","test_assets_read":false}\n' >"${QUEUE_ROOT}/FAILED.json"
    exit 1
  fi
  if (( $(date +%s) >= deadline )); then
    printf '{"status":"failed","reason":"timeout_waiting_for_state_audit","test_assets_read":false}\n' >"${QUEUE_ROOT}/FAILED.json"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

for gpu in "${F0_GPU}" "${F1_GPU}"; do
  while true; do
    active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -z "${active//[[:space:]]/}" ]] && break
    printf '%s GPU %s occupied by %s; retrying in %ss\n' "$(date -Is)" "${gpu}" "${active}" "${POLL_SECONDS}" >>"${QUEUE_ROOT}/queue.log"
    sleep "${POLL_SECONDS}"
  done
done

printf '%s starting selector pilot on F0_GPU=%s F1_GPU=%s\n' "$(date -Is)" "${F0_GPU}" "${F1_GPU}" >>"${QUEUE_ROOT}/queue.log"
env PROJECT_ROOT="${PROJECT_ROOT}" STORAGE_ROOT="${STORAGE_ROOT}" RUN_ROOT="${RUN_ROOT}" \
  F0_GPU="${F0_GPU}" F1_GPU="${F1_GPU}" \
  bash "${PROJECT_ROOT}/scripts/run_sn7_updater_conditioned_selector_pilot.sh" \
  >"${QUEUE_ROOT}/selector_pilot.log" 2>&1

printf '{"status":"complete","stage":"queued_selector_pilot","f0_gpu":%s,"f1_gpu":%s,"test_assets_read":false}\n' \
  "${F0_GPU}" "${F1_GPU}" >"${QUEUE_ROOT}/COMPLETE.json"
