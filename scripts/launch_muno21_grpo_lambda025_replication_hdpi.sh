#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PY=${STORE}/envs/activemap-agent/bin/python
MODEL=/home/wh/hf_models/Qwen3-4B
SFT=${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808
ROLLOUT=${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n512_v1
MATRIX=${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n512_matrix_v1
LOGROOT=${STORE}/logs/muno21_grpo_lambda025_replication_v1
REPORT=${STORE}/reports/muno21_grpo_lambda025_three_seed_20260802
BASELINE=${STORE}/runs/agent/muno21_proxy_grpo_full_g6_lr2p5e7_v1/sft_matched_validation_n512/qwen3_4b_sft_tool_to_belief.jsonl
STATES=${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl
EPISODES=${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl
BELIEF=${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt
SELECTOR=${STORE}/runs/selector
mkdir -p "${MATRIX}" "${LOGROOT}" "${REPORT}"
cd "${PROJECT}"

rollout_args=()
for index in 0 1 2 3 4 5 6 7; do
  seed=$((20260960 + index))
  run=${ROLLOUT}/rollout${index}_seed${seed}
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

train_one() {
  local gpu=$1
  local seed=$2
  local run=${MATRIX}/lambda025_seed${seed}
  CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false PYTHONPATH=src:. \
    "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${SFT}" "${run}" "${rollout_args[@]}" \
    --objective sequence-clip --dynamic-sampling variable-only \
    --false-edit-lagrange 0.25 --false-edit-target 0.0966 \
    --false-edit-dual-lr 1.0 --minimum-keep-trajectories 100 \
    --minimum-false-edit-trajectories 1 --learning-rate 2.5e-7 \
    --entropy-coef 0.001 --gradient-accumulation 4 --seed "${seed}" \
    >"${LOGROOT}/train_seed${seed}.log" 2>&1 &
  echo "$! train GPU${gpu} lambda025 seed${seed}"
}

train_one 0 20260979
train_one 1 20260980
wait

validate_one() {
  local gpu=$1
  local seed=$2
  local run=${MATRIX}/lambda025_seed${seed}
  local output=${run}/validation_full_gpu_selector
  test -f "${run}/COMPLETED"
  CUDA_VISIBLE_DEVICES=${gpu} OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 PYTHONPATH=src:. \
    "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${run}/final" \
    --device cuda:0 --selector-device cuda:0 \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit 512 --max-length 2048 --max-tool-calls 2 --seed "${seed}" \
    >"${LOGROOT}/val_seed${seed}.log" 2>&1 &
  echo "$! validation GPU${gpu} lambda025 seed${seed}"
}

validate_one 0 20260979
validate_one 1 20260980
wait

PYTHONPATH=src:. "${PY}" scripts/summarize_muno21_grpo_v2.py "${REPORT}" \
  --baseline "${BASELINE}" \
  --run lambda025=20260977=${MATRIX}/lambda025_seed20260977/validation_full_gpu_selector/qwen3_4b_sft_tool_to_belief.jsonl \
  --run lambda025=20260979=${MATRIX}/lambda025_seed20260979/validation_full_gpu_selector/qwen3_4b_sft_tool_to_belief.jsonl \
  --run lambda025=20260980=${MATRIX}/lambda025_seed20260980/validation_full_gpu_selector/qwen3_4b_sft_tool_to_belief.jsonl
date -Is >"${REPORT}/PIPELINE_COMPLETED"
