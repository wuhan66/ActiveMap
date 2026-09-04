#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
OOF="${STORAGE_ROOT}/processed/muno21_v2/agent/policy_relative_crossfit_qwen_v1"
TARGETS="${STORAGE_ROOT}/processed/muno21_v2/agent/sequential_controller_qwen_v2_true_balance/train/trajectories.jsonl"
STATES="${STORAGE_ROOT}/processed/muno21_v2/agent/executable_value_v2_noleak/states_train_val_executable_balanced_512.jsonl"
ROLLOUT_ROOT="${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v12_proactive_tool_four_seed_threshold009_v1"
OUTPUT_ROOT="${STORAGE_ROOT}/runs/agent/muno21_oof_terminal_proposal_critic_4seed_v1"
LOG_ROOT="${STORAGE_ROOT}/logs/muno21_oof_terminal_proposal_critic_4seed_v1"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
critic_seeds=(20260930 20260931 20260932 20260933)
agent_seeds=(20260822 20260823 20260824 20260825)
gpus=(1 2 3 4)

for index in "${!critic_seeds[@]}"; do
  critic_seed="${critic_seeds[$index]}"
  agent_seed="${agent_seeds[$index]}"
  gpu="${gpus[$index]}"
  output="${OUTPUT_ROOT}/critic${critic_seed}_agent${agent_seed}"
  log="${LOG_ROOT}/critic${critic_seed}_agent${agent_seed}.log"
  [[ ! -e "${output}" ]] || { echo "refusing to overwrite ${output}" >&2; exit 3; }
  gpu_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${gpu_pids//[[:space:]]/}" ]] || {
    echo "GPU${gpu} has active compute processes: ${gpu_pids}" >&2
    exit 4
  }
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    "${PYTHON}" scripts/train_oof_terminal_proposal_critic.py \
    "${OOF}" "${TARGETS}" "${STATES}" \
    "${ROLLOUT_ROOT}/seed${agent_seed}/edit_conditioned_proactive_tools.jsonl" \
    "${ROLLOUT_ROOT}/seed${agent_seed}/qwen3_4b_sft_calibrated_tool_to_belief.jsonl" \
    "${output}" --seed "${critic_seed}" --device cuda --epochs 100 \
    > "${log}" 2>&1 < /dev/null &
  echo "$!" > "${LOG_ROOT}/critic${critic_seed}_agent${agent_seed}.pid"
  echo "started critic=${critic_seed} agent=${agent_seed} gpu=${gpu} pid=$!"
done
