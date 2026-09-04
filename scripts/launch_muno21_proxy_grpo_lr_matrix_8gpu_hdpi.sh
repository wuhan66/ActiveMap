#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
SFT="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
SWEEP="${STORE}/runs/agent/muno21_grpo_exploration_sweep_v1/t2p0_seed20260842"
GATE="${STORE}/runs/agent/muno21_grpo_g4_temperature_gate_v1"
ROOT="${STORE}/runs/agent/muno21_proxy_grpo_lr_matrix_v1"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
mkdir -p "$ROOT"
cd "$PROJECT"

evaluate_adapter() {
  local gpu="$1" adapter="$2" output="$3" seed="$4"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$adapter" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit 128 --max-length 2048 --max-tool-calls 2 --seed "$seed"
}

train_and_val() {
  local gpu="$1" label="$2" lr="$3" entropy="$4" seed="$5"
  local run="${ROOT}/${label}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=src:. "$PYTHON" scripts/train_recurrent_proxy_grpo.py "$SFT" "$run" \
    --seed "$seed" --learning-rate "$lr" --entropy-coef "$entropy" \
    --epochs 1 --gradient-accumulation 4 --max-length 1536 \
    --rollout "$SWEEP/qwen3_4b_sft_tool_to_belief.jsonl" "$SWEEP/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" \
    --rollout "$GATE/t2p0_seed20260852/qwen3_4b_sft_tool_to_belief.jsonl" "$GATE/t2p0_seed20260852/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" \
    --rollout "$GATE/t2p0_seed20260854/qwen3_4b_sft_tool_to_belief.jsonl" "$GATE/t2p0_seed20260854/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" \
    --rollout "$GATE/t2p0_seed20260856/qwen3_4b_sft_tool_to_belief.jsonl" "$GATE/t2p0_seed20260856/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl" \
    >"${run}.train.log" 2>&1
  evaluate_adapter "$gpu" "$run/final" "$run/validation" "$seed" \
    >"${run}.validation.log" 2>&1
}

train_and_val 0 lr1e7_ent1e3 1e-7 1e-3 20260871 &
train_and_val 1 lr1e7_ent1e3 1e-7 1e-3 20260872 &
train_and_val 2 lr2p5e7_ent1e3 2.5e-7 1e-3 20260873 &
train_and_val 3 lr2p5e7_ent1e3 2.5e-7 1e-3 20260874 &
train_and_val 4 lr5e7_ent1e3 5e-7 1e-3 20260875 &
train_and_val 5 lr5e7_ent1e3 5e-7 1e-3 20260876 &
train_and_val 6 lr1e7_ent0 1e-7 0 20260877 &
evaluate_adapter 7 "$SFT" "${ROOT}/sft_matched_validation" 20260878 \
  >"${ROOT}/sft_matched_validation.log" 2>&1 &
wait
