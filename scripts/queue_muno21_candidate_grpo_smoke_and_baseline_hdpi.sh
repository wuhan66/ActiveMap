#!/usr/bin/env bash
set -euo pipefail

# GPU1: corrected candidate-GRPO smoke.
# GPU3/4/5: matched SFT candidate-policy validation baselines.
# This queue never uses GPUs 0 or 2 and never reads test assets.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807/sft_seed20260897/final}"
SOURCE_ROOT="${SOURCE_ROOT:-${STORE}/runs/agent/muno21_candidate_decoder_reentry_v2_20260807}"
DATA="${DATA:-${STORE}/runs/agent/muno21_post_acquisition_mixed_sft_v1_data_20260807_r1}"
NATIVE_STATES="${NATIVE_STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_candidate_grpo_smoke_baseline_v1_20260807}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_candidate_grpo_smoke_baseline_v1_20260807}"

for path in "${PY}" "${MODEL}/config.json" "${SFT}/adapter_model.safetensors" \
  "${SOURCE_ROOT}/READY_FOR_CANDIDATE_GRPO_SMOKE" "${DATA}/train.jsonl" \
  "${DATA}/val.jsonl" "${NATIVE_STATES}" "${EPISODES}" "${BELIEF}" "${UPDATER}"; do
  [[ -s "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 3; }
done
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || { echo "Queue already complete"; exit 0; }

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
mkdir -p "${ROOT}" "${LOGROOT}"
date -Is >"${ROOT}/QUEUE_STARTED"

rollout_args=()
writeback_args=()
for index in 0 1 2 3; do
  seed=$((20260920 + index))
  run="${SOURCE_ROOT}/candidate_rollouts_g4/rollout${index}_seed${seed}"
  trajectories="${run}/qwen3_4b_sft_tool_to_belief.jsonl"
  calls="${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  writeback="${run}/writeback/writeback.jsonl"
  for path in "${trajectories}" "${calls}" "${writeback}"; do
    [[ -s "${path}" ]] || { echo "Missing rollout input: ${path}" >&2; exit 4; }
  done
  rollout_args+=(--rollout "${trajectories}" "${calls}")
  writeback_args+=(--writeback "${writeback}")
done

run_grpo_smoke() {
  local output="${ROOT}/candidate_grpo_smoke_seed20260940"
  CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${SFT}" "${output}" "${rollout_args[@]}" "${writeback_args[@]}" \
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
    --seed 20260940 >"${LOGROOT}/candidate_grpo_smoke.log" 2>&1

  [[ -s "${output}/COMPLETED" && -s "${output}/final/adapter_model.safetensors" ]] || {
    echo "Candidate GRPO smoke failed to save an adapter" >&2; return 5;
  }

  local replay="${output}/train_native_replay"
  CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${NATIVE_STATES}" "${replay}" --adapter "${output}/final" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --tool-belief-checkpoint "${BELIEF}" --tool-supervision-jsonl "${DATA}/train.jsonl" \
    --max-tool-calls 2 --tool-out-size 256 --split train --oracle-step 1 \
    --budgets 1.5,3.0,4.5 --device cuda:0 --selector-device cpu \
    --seed 20260940 --sample-order seeded-hash --sample-seed 20260882 \
    --limit 16 --do-sample --temperature 1.5 --top-p 0.95 \
    --action-decoder candidate-sample --candidate-score-batch-size 1 \
    --methods qwen3_4b_sft_tool_to_belief \
    >"${LOGROOT}/candidate_grpo_smoke_replay.log" 2>&1
  CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${replay}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${replay}/writeback" --device cuda:0 --split train --image-size 512 \
    --threshold 0.5 --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    >"${LOGROOT}/candidate_grpo_smoke_writeback.log" 2>&1
  date -Is >"${output}/SMOKE_COMPLETED"
}

run_sft_baseline() {
  local gpu="$1" seed="$2"
  local output="${ROOT}/sft_candidate_val_seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${NATIVE_STATES}" "${output}" --adapter "${SFT}" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --tool-belief-checkpoint "${BELIEF}" --tool-supervision-jsonl "${DATA}/val.jsonl" \
    --max-tool-calls 2 --tool-out-size 256 --split val --oracle-step 1 \
    --budgets 1.5,3.0,4.5 --device cuda:0 --selector-device cpu \
    --seed "${seed}" --sample-order seeded-hash --sample-seed 20260882 \
    --do-sample --temperature 1.5 --top-p 0.95 \
    --action-decoder candidate-sample --candidate-score-batch-size 1 \
    --methods qwen3_4b_sft_tool_to_belief \
    >"${LOGROOT}/sft_candidate_val_seed${seed}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${output}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${output}/writeback" --device cuda:0 --split val --image-size 512 \
    --threshold 0.5 --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    >"${LOGROOT}/sft_candidate_val_seed${seed}_writeback.log" 2>&1
  date -Is >"${output}/BASELINE_COMPLETED"
}

run_grpo_smoke & pid1=$!
run_sft_baseline 3 20260941 & pid3=$!
run_sft_baseline 4 20260942 & pid4=$!
run_sft_baseline 5 20260943 & pid5=$!
wait "${pid1}" "${pid3}" "${pid4}" "${pid5}"

date -Is >"${ROOT}/QUEUE_COMPLETED"
echo "candidate GRPO smoke and three matched SFT baselines completed"
