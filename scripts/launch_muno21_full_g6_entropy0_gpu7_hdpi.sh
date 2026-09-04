#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
SFT="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
TRACE="${STORE}/runs/agent/muno21_proxy_grpo_full_g8_t2p0_v1"
OUTPUT="${STORE}/runs/agent/muno21_proxy_grpo_full_g6_lr2p5e7_entropy0_v1/seed20260907"
mkdir -p "$(dirname "$OUTPUT")"
cd "$PROJECT"
args=()
for item in \
  rollout1_seed20260882 rollout2_seed20260883 rollout3_seed20260884 \
  rollout4_seed20260885 rollout5_seed20260886 rollout7_seed20260888; do
  args+=(--rollout "$TRACE/$item/qwen3_4b_sft_tool_to_belief.jsonl" \
                  "$TRACE/$item/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done
CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  PYTHONPATH=src:. "$PYTHON" scripts/train_recurrent_proxy_grpo.py "$SFT" "$OUTPUT" \
  --seed 20260907 --learning-rate 2.5e-7 --entropy-coef 0 \
  --epochs 1 --gradient-accumulation 4 --max-length 1536 "${args[@]}" \
  >"${OUTPUT}.train.log" 2>&1
