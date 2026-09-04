#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
SFT="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
TRACE="${STORE}/runs/agent/muno21_proxy_grpo_full_g8_t2p0_v1"
ROOT="${STORE}/runs/agent/muno21_proxy_grpo_full_g6_lr2p5e7_v1"
EXTRA="${STORE}/runs/agent/muno21_proxy_grpo_full_g8_t2p0_supplement_v1"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
mkdir -p "$ROOT" "$EXTRA" "${STORE}/logs/muno21_proxy_grpo_full_stage2_v1"
cd "$PROJECT"

evaluate_adapter() {
  local gpu="$1" adapter="$2" output="$3" seed="$4" limit="$5"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$adapter" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit "$limit" --max-length 2048 --max-tool-calls 2 --seed "$seed"
}

train_g6_and_val() {
  local gpu="$1" seed="$2"
  local run="${ROOT}/seed${seed}"
  local args=()
  for item in \
    rollout1_seed20260882 rollout2_seed20260883 rollout3_seed20260884 \
    rollout4_seed20260885 rollout5_seed20260886 rollout7_seed20260888; do
    args+=(--rollout "$TRACE/$item/qwen3_4b_sft_tool_to_belief.jsonl" \
                    "$TRACE/$item/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
  done
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH=src:. "$PYTHON" scripts/train_recurrent_proxy_grpo.py "$SFT" "$run" \
    --seed "$seed" --learning-rate 2.5e-7 --entropy-coef 1e-3 \
    --epochs 1 --gradient-accumulation 4 --max-length 1536 "${args[@]}" \
    >"${run}.train.log" 2>&1
  evaluate_adapter "$gpu" "$run/final" "$run/validation" "$seed" 128 \
    >"${run}.validation.log" 2>&1
}

supplement_rollout() {
  local gpu="$1" label="$2" seed="$3"
  local output="${EXTRA}/${label}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$SFT" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split train --budgets 1.5,3.0,4.5 \
    --limit 256 --max-length 2048 --max-tool-calls 2 --seed "$seed" \
    --do-sample --temperature 2.0 --top-p 1.0 --record-training-payload \
    >"${EXTRA}/${label}_seed${seed}.log" 2>&1
}

train_g6_and_val 1 20260891 &
train_g6_and_val 2 20260892 &
train_g6_and_val 3 20260893 &
supplement_rollout 4 supplement0 20260889 &
supplement_rollout 5 supplement1 20260890 &
evaluate_adapter 7 "$SFT" "${ROOT}/sft_matched_validation_n512" 20260894 512 \
  >"${ROOT}/sft_matched_validation_n512.log" 2>&1 &
wait
