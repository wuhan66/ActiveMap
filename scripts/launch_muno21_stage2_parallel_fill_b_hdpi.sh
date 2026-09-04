#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
SFT="${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808"
ROOT="${STORE}/runs/agent/muno21_proxy_grpo_temperature_robustness_t1p8_v1"
BASE="${STORE}/runs/agent/muno21_proxy_grpo_full_g6_lr2p5e7_v1/sft_matched_validation_n512_seed3"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
mkdir -p "$ROOT"
cd "$PROJECT"

run_eval() {
  local gpu="$1" output="$2" split="$3" limit="$4" seed="$5" temperature="$6"
  local sampling=()
  if [[ -n "$temperature" ]]; then
    sampling=(--do-sample --temperature "$temperature" --top-p 1.0 --record-training-payload)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$SFT" --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "$output/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split "$split" --budgets 1.5,3.0,4.5 \
    --limit "$limit" --max-length 2048 --max-tool-calls 2 --seed "$seed" \
    "${sampling[@]}" >"${output}.log" 2>&1
}

run_eval 4 "${ROOT}/rollout0_seed20260898" train 256 20260898 1.8 &
run_eval 5 "${ROOT}/rollout1_seed20260899" train 256 20260899 1.8 &
run_eval 7 "$BASE" val 512 20260900 "" &
wait
