#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
ROOT="${STORE}/runs/agent/muno21_proxy_recurrent_grpo_t2p0_v2"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
cd "$PROJECT"

run_val() {
  local gpu="$1" seed="$2"
  local adapter="${ROOT}/seed${seed}/final"
  local output="${ROOT}/seed${seed}/validation"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$adapter" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit 128 --max-length 2048 --max-tool-calls 2 --seed "$seed" \
    >"${ROOT}/seed${seed}/validation.log" 2>&1
}

run_val 1 20260861 &
run_val 2 20260862 &
run_val 3 20260863 &
wait
