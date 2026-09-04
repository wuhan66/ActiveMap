#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN="${STORE}/runs/sn7_active_catalog"
POLL_SECONDS="${POLL_SECONDS:-120}"

completed() {
  local path="$1/process_result.json"
  [[ -s "${path}" ]] && grep -q '"status": "completed"' "${path}"
}

wait_for() {
  local path="$1"
  until completed "${path}"; do sleep "${POLL_SECONDS}"; done
}

launch_after() {
  local blocker="$1" seed="$2" gpu="$3"
  local output="${RUN}/hybrid_residual_8k_no_tools_fullval_matrix_20260801/seed${seed}"
  wait_for "${blocker}"
  if completed "${output}"; then return 0; fi
  [[ ! -e "${output}" ]] || {
    echo "refusing incomplete output: ${output}" >&2
    return 20
  }
  cd "${PROJECT}"
  bash scripts/run_sn7_hybrid_residual_8k_no_tools_fullval.sh "${seed}" "${gpu}" \
    >"${STORE}/logs/sn7_hybrid_8k_no_tools_seed${seed}_gpu${gpu}_20260801.log" 2>&1
}

launch_after "${RUN}/react_style_fullval_seed20260718" 20260822 7 & p7=$!
launch_after "${RUN}/direct_vlm_no_tools_fullval_seed20260718" 20260823 3 & p3=$!
wait "${p7}"
wait "${p3}"
