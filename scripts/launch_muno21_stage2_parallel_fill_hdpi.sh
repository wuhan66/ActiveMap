#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
SFT="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
ROOT="${STORE}/runs/agent/muno21_proxy_grpo_full_g10_t2p0_supplement_v1"
BASE="${STORE}/runs/agent/muno21_proxy_grpo_full_g6_lr2p5e7_v1/sft_matched_validation_n512_seed2"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
mkdir -p "$ROOT"
cd "$PROJECT"

common=(
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt"
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt"
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt"
  --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" --tool-out-size 256
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}"
  --methods qwen3_4b_sft_tool_to_belief --max-length 2048 --max-tool-calls 2
)

rollout() {
  local gpu="$1"
  local label="$2"
  local seed="$3"
  local output="${ROOT}/${label}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$SFT" --device cuda:0 --selector-device cpu \
    "${common[@]}" --tool-artifact-root "$output/tool_artifacts" \
    --split train --budgets 1.5,3.0,4.5 --limit 256 --seed "$seed" \
    --do-sample --temperature 2.0 --top-p 1.0 --record-training-payload \
    >"${output}.log" 2>&1
}

rollout 4 supplement2 20260896 &
rollout 5 supplement3 20260897 &
if [[ ! -e "$BASE" ]]; then
  CUDA_VISIBLE_DEVICES=7 PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$BASE" --adapter "$SFT" --device cuda:0 --selector-device cpu \
    "${common[@]}" --tool-artifact-root "$BASE/tool_artifacts" \
    --split val --budgets 1.5,3.0,4.5 --limit 512 --seed 20260895 \
    >"${BASE}.log" 2>&1 &
fi
wait
