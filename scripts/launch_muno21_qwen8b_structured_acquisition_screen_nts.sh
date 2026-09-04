#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
AGENT_ENV="${ACTIVEMAP_AGENT_ENV:-/home/wh/venvs/activemap}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-8B}"
ADAPTER="${MUNO21_V12_ADAPTER:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_8b_balanced_tool_sft_seed20260840/checkpoints/checkpoint-400}"
SELECTOR_SEED="${MUNO21_SELECTOR_SEED:-20260811}"
RUN_ROOT="${MUNO21_SCREEN_ROOT:-${STORAGE_ROOT}/artifacts/paper_rollouts/qwen3_8b_structured_acquisition_matched_v2_threshold009}"
LOG_ROOT="${STORAGE_ROOT}/logs"
LOG_TAG="${MUNO21_SCREEN_LOG_TAG:-seed20260840}"
GPU_NO_BELIEF="${GPU_NO_BELIEF:-1}"
GPU_RECURRENT="${GPU_RECURRENT:-2}"
TOOL_NEED_THRESHOLD="${MUNO21_V12_TOOL_NEED_THRESHOLD:-0.09}"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

run_variant() {
  local gpu="$1"
  local method="$2"
  local output="$3"
  env \
    ACTIVEMAP_PROJECT_ROOT="${PROJECT_ROOT}" \
    ACTIVEMAP_STORAGE_ROOT="${STORAGE_ROOT}" \
    ACTIVEMAP_AGENT_ENV="${AGENT_ENV}" \
    MUNO21_V12_GPU="${gpu}" \
    MUNO21_AGENT_SEED=20260840 \
    MUNO21_SELECTOR_SEED="${SELECTOR_SEED}" \
    MUNO21_AGENT_MODEL="${MODEL}" \
    MUNO21_V12_ADAPTER="${ADAPTER}" \
    MUNO21_V12_METHODS="${method}" \
    MUNO21_V12_TOOL_NEED_THRESHOLD="${TOOL_NEED_THRESHOLD}" \
    MUNO21_V12_ASSESS=0 \
    MUNO21_V12_ASSET_ROOT_MAP="/home/wh/ActiveMap=${STORAGE_ROOT}" \
    MUNO21_V12_ROLLOUT_ROOT="${output}" \
    bash "${PROJECT_ROOT}/scripts/run_muno21_v12_proactive_rollout.sh"
}

run_variant \
  "${GPU_NO_BELIEF}" \
  qwen3_4b_sft_calibrated_tools_no_belief \
  "${RUN_ROOT}/structured_gate_identity_belief_seed20260840" \
  >"${LOG_ROOT}/qwen3_8b_structured_gate_identity_${LOG_TAG}.log" 2>&1 &
pid_no_belief=$!

run_variant \
  "${GPU_RECURRENT}" \
  qwen3_4b_sft_calibrated_tool_to_belief \
  "${RUN_ROOT}/structured_gate_recurrent_belief_seed20260840" \
  >"${LOG_ROOT}/qwen3_8b_structured_gate_recurrent_${LOG_TAG}.log" 2>&1 &
pid_recurrent=$!

status=0
wait "${pid_no_belief}" || status=1
wait "${pid_recurrent}" || status=1
if [[ "${status}" -eq 0 ]]; then
  touch "${RUN_ROOT}/MATRIX_COMPLETED"
fi
exit "${status}"
