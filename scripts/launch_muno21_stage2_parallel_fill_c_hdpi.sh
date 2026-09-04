#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
SFT="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
ROOT18="${STORE}/runs/agent/muno21_proxy_grpo_temperature_robustness_t1p8_v1"
ROOT22="${STORE}/runs/agent/muno21_proxy_grpo_temperature_robustness_t2p2_v1"
mkdir -p "$ROOT18" "$ROOT22"
cd "$PROJECT"

rollout() {
  local gpu="$1" output="$2" seed="$3" temperature="$4"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$SFT" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "$output/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split train --budgets 1.5,3.0,4.5 \
    --limit 256 --max-length 2048 --max-tool-calls 2 --seed "$seed" \
    --do-sample --temperature "$temperature" --top-p 1.0 --record-training-payload \
    >"${output}.log" 2>&1
}

rollout 1 "${ROOT18}/rollout2_seed20260901" 20260901 1.8 &
rollout 2 "${ROOT18}/rollout3_seed20260902" 20260902 1.8 &
rollout 3 "${ROOT22}/rollout0_seed20260903" 20260903 2.2 &
rollout 4 "${ROOT22}/rollout1_seed20260904" 20260904 2.2 &
rollout 5 "${ROOT22}/rollout2_seed20260905" 20260905 2.2 &
rollout 7 "${ROOT22}/rollout3_seed20260906" 20260906 2.2 &
wait
