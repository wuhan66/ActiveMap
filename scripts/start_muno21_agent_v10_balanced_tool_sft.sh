#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
if [[ -n "${ACTIVEMAP_SERVER_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${ACTIVEMAP_SERVER_ENV}"
fi
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
SEED="${MUNO21_AGENT_SEED:-20260821}"

export MUNO21_AGENT_DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools}"
export MUNO21_AGENT_RUN_DIR="${MUNO21_AGENT_RUN_DIR:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed${SEED}}"
export MUNO21_AGENT_PROTOCOL="balanced-opportunity-sparse-grounded-tool-sft-v2-full-budget"
export MUNO21_AGENT_SEED="${SEED}"
export MUNO21_AGENT_EARLY_STOPPING_PATIENCE="${MUNO21_AGENT_EARLY_STOPPING_PATIENCE:-0}"
export MUNO21_AGENT_SAVE_TOTAL_LIMIT="${MUNO21_AGENT_SAVE_TOTAL_LIMIT:-8}"

[[ -s "${MUNO21_AGENT_DATA_ROOT}/balanced_prior_audit.json" ]] || {
  echo "Missing balanced-prior audit: ${MUNO21_AGENT_DATA_ROOT}/balanced_prior_audit.json" >&2
  exit 1
}
exec bash "${PROJECT_ROOT}/scripts/start_muno21_agent_sparse_tool_sft.sh"
