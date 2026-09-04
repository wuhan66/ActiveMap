#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
ADAPTER="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
SWEEP="${STORE}/runs/agent/muno21_grpo_exploration_sweep_v1/t2p0_seed20260842"
GATE="${STORE}/runs/agent/muno21_grpo_g4_temperature_gate_v1"
ROOT="${STORE}/runs/agent/muno21_proxy_recurrent_grpo_t2p0_v2"
mkdir -p "$ROOT"
cd "$PROJECT"

run_seed() {
  local gpu="$1" seed="$2"
  local output="${ROOT}/seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=src:. "$PYTHON" \
    scripts/train_recurrent_proxy_grpo.py "$ADAPTER" "$output" \
    --seed "$seed" --learning-rate 1e-6 --epochs 1 --gradient-accumulation 4 \
    --max-length 1536 \
    --rollout "$SWEEP/qwen3_4b_sft_tool_to_belief.jsonl" "$SWEEP/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" \
    --rollout "$GATE/t2p0_seed20260852/qwen3_4b_sft_tool_to_belief.jsonl" "$GATE/t2p0_seed20260852/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" \
    --rollout "$GATE/t2p0_seed20260854/qwen3_4b_sft_tool_to_belief.jsonl" "$GATE/t2p0_seed20260854/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" \
    --rollout "$GATE/t2p0_seed20260856/qwen3_4b_sft_tool_to_belief.jsonl" "$GATE/t2p0_seed20260856/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" \
    >"${output}.log" 2>&1
}

run_seed 1 20260861 &
run_seed 2 20260862 &
run_seed 3 20260863 &
wait
