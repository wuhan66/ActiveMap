#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PY=${STORE}/envs/activemap-agent/bin/python
MODEL=/home/wh/hf_models/Qwen3-4B
SFT=${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808
STATES=${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl
EPISODES=${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl
BELIEF=${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt
SELECTOR=${STORE}/runs/selector
ROOT=${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n512_v1
mkdir -p "${ROOT}"
cd "${PROJECT}"

rollout() {
  local gpu=$1
  local index=$2
  local seed=$3
  local output=${ROOT}/rollout${index}_seed${seed}
  CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=src:. \
    "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${SFT}" \
    --device cuda:0 --selector-device cuda:0 \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split train --budgets 1.5,3.0,4.5 \
    --limit 512 --sample-order seeded-hash --sample-seed 20260960 \
    --max-length 2048 --max-tool-calls 2 --seed "${seed}" \
    --do-sample --temperature 2.2 --top-p 1.0 --record-training-payload \
    >"${output}.log" 2>&1 &
  echo "$! GPU${gpu} rollout${index} seed${seed}"
}

for gpu in 0 1 2 3 4 5 6 7; do
  rollout "${gpu}" "${gpu}" "$((20260960 + gpu))"
done
wait
