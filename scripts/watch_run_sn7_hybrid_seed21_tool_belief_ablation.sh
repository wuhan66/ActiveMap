#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
SEED=20260821
GPU=5
POLL_SECONDS="${POLL_SECONDS:-120}"
MAIN="${RUN}/hybrid_residual_8k_fullval_matrix_20260801/seed${SEED}/evaluation/traces.jsonl"

completed() {
  local result="$1/process_result.json"
  [[ -s "${result}" ]] && grep -q '"status": "completed"' "${result}"
}

wait_for() {
  until completed "$1"; do sleep "${POLL_SECONDS}"; done
}

wait_for "${RUN}/hybrid_residual_8k_no_tools_fullval_matrix_20260801/seed${SEED}"
cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
for variant in frozen_prior forced_tools; do
  output="${RUN}/hybrid_residual_8k_tool_belief_ablation_20260801/${variant}/seed${SEED}"
  if ! completed "${output}"; then
    bash scripts/run_sn7_hybrid_residual_8k_tool_belief_ablation.sh \
      "${SEED}" "${GPU}" "${variant}" \
      >"${STORE}/logs/sn7_hybrid_8k_${variant}_seed${SEED}_gpu${GPU}_20260801.log" 2>&1
  fi
  comparison="${output}/main_minus_${variant}.json"
  if [[ ! -e "${comparison}" ]]; then
    "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
      "${comparison}" \
      --records "${variant}=${output}/evaluation/traces.jsonl" \
      --records "main=${MAIN}" --candidate main \
      --repetitions 10000 --seed "${SEED}"
  fi
done
