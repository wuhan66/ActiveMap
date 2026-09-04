#!/usr/bin/env bash
set -euo pipefail

# Convert the four train-only recurrent rollouts into executable map rewards.
# This is a data-preparation stage; it does not update the language model.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORE}/envs/activemap-agent/bin/python}"
GPU="${GPU:-4}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_recurrent_grpo_stage1_g4_v4}"

rollouts=(
  "${ROOT}/rollout0_seed20260831/qwen3_4b_sft_tool_to_belief.jsonl"
  "${ROOT}/rollout1_seed20260832/qwen3_4b_sft_tool_to_belief.jsonl"
  "${ROOT}/rollout2_seed20260833/qwen3_4b_sft_tool_to_belief.jsonl"
  "${ROOT}/rollout3_seed20260834/qwen3_4b_sft_tool_to_belief.jsonl"
)

cd "${PROJECT}"
for index in "${!rollouts[@]}"; do
  rollout="${rollouts[$index]}"
  output="${ROOT}/rollout${index}_writeback"
  if [[ -s "${output}/summary.json" ]]; then
    echo "writeback already complete: ${output}"
    continue
  fi
  [[ -s "${rollout}" ]] || {
    echo "missing rollout: ${rollout}" >&2
    exit 2
  }
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT}/src:." "${PYTHON}" \
    scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${rollout}" "${output}" \
    --device cuda:0 --split train --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}"
done

echo "recurrent executable writeback preparation complete"
