#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
ROOT="${SN7_SEQUENTIAL_ROOT:-${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1}"
WAIT_SECONDS="${WAIT_SECONDS:-60}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-21600}"
STARTED="${SECONDS}"

required=(
  "${ROOT}/selector_states_train_val.jsonl"
  "${ROOT}/selector_states_train_val.merge_summary.json"
  "${ROOT}/selector_states_train_val.audit.json"
  "${ROOT}/selector_states_train_val.utility_audit.json"
)

while true; do
  ready=true
  for path in "${required[@]}"; do
    [[ -s "${path}" ]] || ready=false
  done
  if [[ "${ready}" == true ]]; then
    exec bash "${PROJECT_ROOT}/scripts/prepare_sn7_sequential_selector.sh" full_sft
  fi
  if (( SECONDS - STARTED >= MAX_WAIT_SECONDS )); then
    echo "Timed out waiting for audited full selector merge" >&2
    exit 7
  fi
  printf '%(%F %T)T waiting for audited selector merge\n' -1
  sleep "${WAIT_SECONDS}"
done
