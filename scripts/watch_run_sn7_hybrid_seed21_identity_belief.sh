#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
SEED=20260821
GPU=5
POLL_SECONDS="${POLL_SECONDS:-120}"
FORCED="${RUN}/hybrid_residual_8k_tool_belief_ablation_20260801/forced_tools/seed${SEED}"
OUTPUT="${RUN}/hybrid_residual_8k_tool_belief_ablation_20260801/identity/seed${SEED}"
MAIN="${RUN}/hybrid_residual_8k_fullval_matrix_20260801/seed${SEED}/evaluation/traces.jsonl"

completed() {
  local result="$1/process_result.json"
  [[ -s "${result}" ]] && grep -q '"status": "completed"' "${result}"
}

until completed "${FORCED}"; do sleep "${POLL_SECONDS}"; done
cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
if ! completed "${OUTPUT}"; then
  bash scripts/run_sn7_hybrid_residual_8k_tool_belief_ablation.sh \
    "${SEED}" "${GPU}" identity \
    >"${STORE}/logs/sn7_hybrid_8k_identity_seed${SEED}_gpu${GPU}_20260801.log" 2>&1
fi

COMPARISON="${OUTPUT}/main_minus_identity.json"
if [[ ! -e "${COMPARISON}" ]]; then
  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${COMPARISON}" \
    --records "identity=${OUTPUT}/evaluation/traces.jsonl" \
    --records "main=${MAIN}" --candidate main \
    --repetitions 10000 --seed "${SEED}"
fi
