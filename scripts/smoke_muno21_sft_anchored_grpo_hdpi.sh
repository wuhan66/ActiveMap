#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
SFT="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
SFT_REPLAY="${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.jsonl"
ROLLOUT_ROOT="${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n1024_v1"
OUTPUT="${STORE}/runs/agent/muno21_sft_anchored_grpo_smoke_v3"

[[ ! -e "${OUTPUT}" ]] || exit 0
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"

rollout_args=()
for index in 0 1 2 3; do
  seed=$((20261020 + index))
  run="${ROLLOUT_ROOT}/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/train_recurrent_proxy_grpo.py \
  "${SFT}" "${OUTPUT}" "${rollout_args[@]}" \
  --objective sequence-clip --dynamic-sampling variable-only --max-groups 64 \
  --false-edit-lagrange 0.25 --false-edit-target 0.0966 \
  --false-edit-dual-lr 1.0 --learning-rate 2.5e-7 --entropy-coef 0.001 \
  --sequence-kl-coef 0.01 --sft-replay "${SFT_REPLAY}" \
  --sft-replay-weight 0.10 --sft-replay-probability 1.0 \
  --sft-tool-replay-fraction 0.50 \
  --gradient-accumulation 2 --epochs 1 --seed 20261200
