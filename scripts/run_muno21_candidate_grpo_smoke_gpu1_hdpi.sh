#!/usr/bin/env bash
set -euo pipefail

# Retry of the corrected categorical candidate-policy smoke after the
# completion-only LM projection memory fix. Uses GPU1 and train-only data.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807/sft_seed20260897/final}"
SOURCE_ROOT="${SOURCE_ROOT:-${STORE}/runs/agent/muno21_candidate_decoder_reentry_v2_20260807}"
DATA="${DATA:-${STORE}/runs/agent/muno21_post_acquisition_mixed_sft_v1_data_20260807_r1}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_candidate_grpo_smoke_baseline_v1_20260807/candidate_grpo_smoke_seed20260940_v2}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_candidate_grpo_smoke_baseline_v1_20260807}"

[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing output: ${OUTPUT}" >&2; exit 2; }
mkdir -p "${LOGROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

rollout_args=()
writeback_args=()
for index in 0 1 2 3; do
  seed=$((20260920 + index))
  run="${SOURCE_ROOT}/candidate_rollouts_g4/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
  writeback_args+=(--writeback "${run}/writeback/writeback.jsonl")
done

CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/train_recurrent_proxy_grpo.py \
  "${SFT}" "${OUTPUT}" "${rollout_args[@]}" "${writeback_args[@]}" \
  --reward-mode executable --objective candidate-clip \
  --candidate-score-batch-size 1 --dynamic-sampling variable-only \
  --max-groups 12 --minimum-reward-std 1e-6 \
  --minimum-keep-trajectories 2 --minimum-commit-trajectories 4 \
  --minimum-tool-trajectories 4 --minimum-false-edit-trajectories 2 \
  --false-edit-lagrange 0.25 --false-edit-target 0.10 --false-edit-dual-lr 0.5 \
  --learning-rate 2.5e-7 --epochs 1 --gradient-accumulation 2 \
  --entropy-coef 0.001 --sequence-kl-coef 0.01 \
  --sft-replay "${DATA}/train.jsonl" --sft-replay-weight 0.10 \
  --sft-replay-probability 0.50 --sft-tool-replay-fraction 0.50 \
  --seed 20260940 >"${LOGROOT}/candidate_grpo_smoke_v2.log" 2>&1

[[ -s "${OUTPUT}/COMPLETED" && -s "${OUTPUT}/final/adapter_model.safetensors" ]]
REPLAY="${OUTPUT}/train_native_replay"
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/evaluate_agent_rollouts.py \
  "${MODEL}" "${STATES}" "${REPLAY}" --adapter "${OUTPUT}/final" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
  --episodes "${EPISODES}" --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
  --tool-belief-checkpoint "${BELIEF}" --tool-supervision-jsonl "${DATA}/train.jsonl" \
  --max-tool-calls 2 --tool-out-size 256 --split train --oracle-step 1 \
  --budgets 1.5,3.0,4.5 --device cuda:0 --selector-device cpu \
  --seed 20260940 --sample-order seeded-hash --sample-seed 20260882 --limit 16 \
  --do-sample --temperature 1.5 --top-p 0.95 --action-decoder candidate-sample \
  --candidate-score-batch-size 1 --methods qwen3_4b_sft_tool_to_belief \
  >"${LOGROOT}/candidate_grpo_smoke_v2_replay.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/evaluate_agent_map_writeback.py \
  "${UPDATER}" "${EPISODES}" "${REPLAY}/qwen3_4b_sft_tool_to_belief.jsonl" \
  "${REPLAY}/writeback" --device cuda:0 --split train --image-size 512 \
  --threshold 0.5 --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
  >"${LOGROOT}/candidate_grpo_smoke_v2_writeback.log" 2>&1
date -Is >"${OUTPUT}/SMOKE_COMPLETED"
