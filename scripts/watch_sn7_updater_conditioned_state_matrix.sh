#!/usr/bin/env bash
# Wait for the immutable C5 state cache, then run only its fail-closed audit.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:?set the v3 run root}"
POLL_SECONDS="${POLL_SECONDS:-120}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-43200}"

[[ "${POLL_SECONDS}" =~ ^[1-9][0-9]*$ ]] || { echo 'invalid POLL_SECONDS' >&2; exit 2; }
[[ "${TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || { echo 'invalid TIMEOUT_SECONDS' >&2; exit 2; }

deadline=$(( $(date +%s) + TIMEOUT_SECONDS ))
states=("f0 train" "f0 val" "f1 train" "f1 val")
while true; do
  complete=0
  for item in "${states[@]}"; do
    read -r state split <<< "${item}"
    root="${RUN_ROOT}/states/${state}/${split}"
    if [[ -s "${root}/FAILED.json" ]]; then
      printf '{"status":"failed","reason":"state_shard_failed","state":"%s","split":"%s","test_assets_read":false}\n' \
        "${state}" "${split}" > "${RUN_ROOT}/STATE_AUDIT_FAILED.json"
      exit 1
    fi
    [[ -s "${root}/COMPLETE.json" && -s "${root}/states.jsonl" ]] && complete=$((complete + 1))
  done
  [[ "${complete}" -eq 4 ]] && break
  if (( $(date +%s) >= deadline )); then
    printf '{"status":"failed","reason":"timeout_waiting_for_state_cache","test_assets_read":false}\n' \
      > "${RUN_ROOT}/STATE_AUDIT_FAILED.json"
    exit 1
  fi
  sleep "${POLL_SECONDS}"
done

cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/audit_policy_relative_staleness.py \
  --old-traces "${RUN_ROOT}/states/f0/train/states.jsonl" \
  --new-traces "${RUN_ROOT}/states/f1/train/states.jsonl" \
  --oracle-step 0 --output-dir "${RUN_ROOT}/audits/train_step0"
"${PYTHON}" scripts/audit_policy_relative_staleness.py \
  --old-traces "${RUN_ROOT}/states/f0/val/states.jsonl" \
  --new-traces "${RUN_ROOT}/states/f1/val/states.jsonl" \
  --oracle-step 0 --output-dir "${RUN_ROOT}/audits/val_step0"

printf '{"status":"complete","stage":"matched_step0_staleness_audit","test_assets_read":false}\n' \
  > "${RUN_ROOT}/STATE_AUDIT_COMPLETE.json"
