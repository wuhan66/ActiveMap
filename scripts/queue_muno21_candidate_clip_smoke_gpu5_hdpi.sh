#!/usr/bin/env bash
set -euo pipefail

# Small train-only candidate-policy update after exact behavior/update parity.
# This is an optimization smoke, not a promotion or validation experiment.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
GPU="${GPU:-5}"
SOURCE="${SOURCE:-${STORE}/runs/agent/muno21_candidate_token_exact_gpu5_20260808}"
PARITY="${PARITY:-${STORE}/runs/agent/muno21_candidate_token_exact_parity_gpu5_20260808}"
ADAPTER="${ADAPTER:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807/sft_seed20260897/final}"
SFT="${SFT:-${STORE}/runs/agent/muno21_post_acquisition_mixed_sft_v1_data_20260807_r1/train.jsonl}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_candidate_clip_smoke_gpu5_20260808}"
LOG="${LOG:-${STORE}/logs/muno21_candidate_clip_smoke_gpu5_20260808.log}"

[[ "${GPU}" != "0" && "${GPU}" != "2" ]] || {
  echo "GPU ${GPU} is excluded" >&2
  exit 3
}
test -s "${PARITY}/READY_FOR_CONSTRAINED_GRPO_SMOKE"
test -s "${SFT}"
[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing existing smoke output: ${OUTPUT}" >&2
  exit 22
}

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
rollout_args=()
writeback_args=()
seeds=(20260930 20260931 20260932 20260933)
for index in 0 1 2 3; do
  seed="${seeds[$index]}"
  root="${SOURCE}/candidate_rollouts_g4/rollout${index}_seed${seed}"
  rollout_args+=(
    --rollout
    "${root}/qwen3_4b_sft_tool_to_belief.jsonl"
    "${root}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  )
  writeback_args+=(--writeback "${root}/writeback/writeback.jsonl")
done

mkdir -p "$(dirname "${LOG}")"
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/train_recurrent_proxy_grpo.py \
  "${ADAPTER}" "${OUTPUT}" \
  "${rollout_args[@]}" "${writeback_args[@]}" \
  --reward-mode executable --objective candidate-clip \
  --candidate-score-batch-size 1 --dynamic-sampling variable-only \
  --max-groups 8 --epochs 1 --learning-rate 2e-7 \
  --gradient-accumulation 8 --max-length 8172 \
  --clip-epsilon-low 0.10 --clip-epsilon-high 0.10 \
  --entropy-coef 0.002 --sequence-kl-coef 0.05 \
  --false-edit-lagrange 0.50 --false-edit-target 0.10 \
  --false-edit-dual-lr 1.0 \
  --minimum-keep-trajectories 2 --minimum-commit-trajectories 8 \
  --minimum-tool-trajectories 8 \
  --sft-replay "${SFT}" --sft-replay-weight 0.05 \
  --sft-replay-probability 0.50 --sft-tool-replay-fraction 0.50 \
  --seed 20260940 >"${LOG}" 2>&1

test -s "${OUTPUT}/COMPLETED"
cp "${OUTPUT}/COMPLETED" "${OUTPUT}/SMOKE_COMPLETED_NO_PROMOTION"
echo "candidate-clip train-only smoke completed; validation is not authorized here"
