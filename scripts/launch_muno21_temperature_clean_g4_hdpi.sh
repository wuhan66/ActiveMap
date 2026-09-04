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
ROOT18="${STORE}/runs/agent/muno21_temperature_t1p8_clean_g4_v1"
ROOT22="${STORE}/runs/agent/muno21_temperature_t2p2_clean_g4_v1"
mkdir -p "$ROOT18" "$ROOT22"
cd "$PROJECT"

rollout() {
  local gpu="$1" root="$2" index="$3" seed="$4" temperature="$5"
  local output="${root}/rollout${index}_seed${seed}"
  CUDA_VISIBLE_DEVICES="$gpu" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=src:. \
    "$PYTHON" scripts/evaluate_agent_rollouts.py \
    "$MODEL" "$STATES" "$output" --adapter "$SFT" \
    --device cuda:0 --selector-device cuda:0 \
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

(rollout 1 "$ROOT18" 0 20260920 1.8; rollout 1 "$ROOT18" 3 20260923 1.8) &
rollout 2 "$ROOT18" 1 20260921 1.8 &
rollout 3 "$ROOT18" 2 20260922 1.8 &
(rollout 4 "$ROOT22" 0 20260924 2.2; rollout 4 "$ROOT22" 3 20260927 2.2) &
rollout 5 "$ROOT22" 1 20260925 2.2 &
rollout 7 "$ROOT22" 2 20260926 2.2 &
wait
