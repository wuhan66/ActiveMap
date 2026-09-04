#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
OOF="${STORAGE_ROOT}/processed/muno21_v2/agent/policy_relative_crossfit_qwen_v1"
GENERATED="${STORAGE_ROOT}/runs/agent_eval/qwen3_vl_4b_terminal_oof_predictions_v1"
STATES="${STORAGE_ROOT}/processed/muno21_v2/agent/executable_value_v2_noleak/states_train_val_executable_balanced_512.jsonl"
ROLLOUT_ROOT="${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v12_proactive_tool_four_seed_threshold009_v1"
OUTPUT="${STORAGE_ROOT}/runs/agent/muno21_residual_risk_terminal_scorer_seed20261020_v3"
LOG="${STORAGE_ROOT}/logs/muno21_residual_risk_terminal_scorer_seed20261020_v3.log"
GPU="${GPU:-5}"

[[ "${GPU}" =~ ^(1|3|4|5)$ ]] || { echo "GPU must be one of 1/3/4/5" >&2; exit 4; }
[[ ! -e "${OUTPUT}" ]] || { echo "refusing to overwrite ${OUTPUT}" >&2; exit 3; }
for fold in 0 1 2; do
  [[ -s "${GENERATED}/fold${fold}/predictions.jsonl" ]] || exit 3
done
gpu_pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${gpu_pids//[[:space:]]/}" ]] || {
  echo "GPU${GPU} occupied: ${gpu_pids}" >&2
  exit 4
}

mkdir -p "$(dirname "${LOG}")"
cd "${PROJECT_ROOT}"
nohup env CUDA_VISIBLE_DEVICES="${GPU}" PYTHONUNBUFFERED=1 \
  "${PYTHON}" scripts/train_residual_risk_terminal_scorer.py \
  "${OOF}" "${GENERATED}" "${STATES}" \
  "${ROLLOUT_ROOT}/seed20260822/edit_conditioned_proactive_tools.jsonl" \
  "${ROLLOUT_ROOT}/seed20260822/qwen3_4b_sft_calibrated_tool_to_belief.jsonl" \
  "${OUTPUT}" --seed 20261020 --device cuda \
  > "${LOG}" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${LOG}.pid"
echo "started residual-risk pilot gpu=${GPU} pid=${pid} output=${OUTPUT}"
