#!/usr/bin/env bash
set -euo pipefail
PROJECT="/home/wh/projects/activemap-v1"
STORE="/home/wh/ActiveMap"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-4B"
G8="${STORE}/runs/agent/muno21_proxy_grpo_full_g8_lr2p5e7_v1"
ENT0="${STORE}/runs/agent/muno21_proxy_grpo_full_g6_lr2p5e7_entropy0_v1"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
cd "$PROJECT"

validate() {
  local gpu="$1" run="$2" seed="$3"
  local output="${run}/validation_full_gpu_selector"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=src:. \
    "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$run/final" \
    --device cuda:0 --selector-device cuda:0 \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "$EPISODES" --tool-belief-checkpoint "$BELIEF" \
    --tool-artifact-root "$output/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit 512 --max-length 2048 --max-tool-calls 2 --seed "$seed" \
    >"${run}/validation_full_gpu_selector.log" 2>&1
}

validate 1 "$G8/seed20260908" 20260908 &
validate 2 "$G8/seed20260909" 20260909 &
validate 3 "$G8/seed20260910" 20260910 &
validate 4 "$ENT0/seed20260907" 20260907 &
validate 5 "$ENT0/seed20260911" 20260911 &
validate 7 "$ENT0/seed20260912" 20260912 &
wait
