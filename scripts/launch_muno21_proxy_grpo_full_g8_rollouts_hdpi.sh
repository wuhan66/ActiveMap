#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
ADAPTER="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
ROOT="${STORE}/runs/agent/muno21_proxy_grpo_full_g8_t2p0_v1"
LOGS="${STORE}/logs/muno21_proxy_grpo_full_g8_t2p0_v1"
mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_actor() {
  local gpu="$1" rollout="$2" seed="$3"
  local output="${ROOT}/rollout${rollout}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$ADAPTER" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split train --budgets 1.5,3.0,4.5 \
    --limit 256 --max-length 2048 --max-tool-calls 2 --seed "$seed" \
    --do-sample --temperature 2.0 --top-p 1.0 --record-training-payload \
    >"${LOGS}/rollout${rollout}_seed${seed}.log" 2>&1
}

for gpu in 0 1 2 3 4 5 6 7; do
  run_actor "$gpu" "$gpu" "$((20260881 + gpu))" &
done
wait
