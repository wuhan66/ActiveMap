#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORE="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
MODEL="/home/wh/hf_models/Qwen3-8B"
ADAPTER="${STORE}/runs/agent/muno21_qwen3_8b_balanced_tool_sft_seed20260840/checkpoints/checkpoint-400"
ROOT="${STORE}/artifacts/paper_rollouts/qwen3_8b_gate_threshold_frontier_v1"
LOG_ROOT="${STORE}/logs/qwen3_8b_gate_threshold_frontier_v1"

mkdir -p "${ROOT}" "${LOG_ROOT}"
exec 9>"${ROOT}/.launcher.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/MATRIX_COMPLETED" ]] || exit 0

run_threshold() {
  local gpu="$1" threshold="$2"
  local label="threshold_${threshold/./p}"
  env \
    ACTIVEMAP_PROJECT_ROOT="${PROJECT_ROOT}" \
    ACTIVEMAP_STORAGE_ROOT="${STORE}" \
    ACTIVEMAP_AGENT_ENV=/home/wh/venvs/activemap \
    MUNO21_V12_GPU="${gpu}" MUNO21_AGENT_SEED=20260840 \
    MUNO21_SELECTOR_SEED=20260811 MUNO21_AGENT_MODEL="${MODEL}" \
    MUNO21_V12_ADAPTER="${ADAPTER}" \
    MUNO21_V12_METHODS=qwen3_4b_sft_calibrated_tool_to_belief \
    MUNO21_V12_TOOL_NEED_THRESHOLD="${threshold}" MUNO21_V12_ASSESS=0 \
    MUNO21_V12_ASSET_ROOT_MAP="/home/wh/ActiveMap=${STORE}" \
    MUNO21_V12_ROLLOUT_ROOT="${ROOT}/${label}" \
    bash "${PROJECT_ROOT}/scripts/run_muno21_v12_proactive_rollout.sh" \
    >"${LOG_ROOT}/${label}.log" 2>&1
}

(run_threshold 1 0.03 && run_threshold 1 0.05) &
pid_low=$!
(run_threshold 2 0.15 && run_threshold 2 0.30) &
pid_high=$!
status=0
wait "${pid_low}" || status=1
wait "${pid_high}" || status=1
if [[ "${status}" -eq 0 ]]; then
  touch "${ROOT}/MATRIX_COMPLETED"
else
  touch "${ROOT}/MATRIX_FAILED"
fi
exit "${status}"
