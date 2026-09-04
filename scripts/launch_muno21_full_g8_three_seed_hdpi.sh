#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
SFT="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
TRACE="${STORE}/runs/agent/muno21_proxy_grpo_full_g8_t2p0_v1"
EXTRA="${STORE}/runs/agent/muno21_proxy_grpo_full_g8_t2p0_supplement_v1"
ROOT="${STORE}/runs/agent/muno21_proxy_grpo_full_g8_lr2p5e7_v1"
mkdir -p "$ROOT"
cd "$PROJECT"

train_seed() {
  local gpu="$1" seed="$2" output="${ROOT}/seed${2}"
  local args=()
  for item in \
    rollout1_seed20260882 rollout2_seed20260883 rollout3_seed20260884 \
    rollout4_seed20260885 rollout5_seed20260886 rollout7_seed20260888; do
    args+=(--rollout "$TRACE/$item/qwen3_4b_sft_tool_to_belief.jsonl" \
                    "$TRACE/$item/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
  done
  for item in supplement0_seed20260889 supplement1_seed20260890; do
    args+=(--rollout "$EXTRA/$item/qwen3_4b_sft_tool_to_belief.jsonl" \
                    "$EXTRA/$item/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
  done
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=src:. "$PYTHON" scripts/train_recurrent_proxy_grpo.py "$SFT" "$output" \
    --seed "$seed" --learning-rate 2.5e-7 --entropy-coef 1e-3 \
    --epochs 1 --gradient-accumulation 4 --max-length 1536 "${args[@]}" \
    >"${output}.train.log" 2>&1
}

train_seed 1 20260908 &
train_seed 2 20260909 &
train_seed 3 20260910 &
wait
