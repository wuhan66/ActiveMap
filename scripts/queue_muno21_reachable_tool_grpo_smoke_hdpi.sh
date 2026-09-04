#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT_ADAPTER="${SFT_ADAPTER:-${STORE}/runs/agent/muno21_qwen3_4b_reachable_tool_support_sft_seed20260831/final}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_rollouts_v1}"
AUDIT="${AUDIT:-${ROOT}/diversity_audit.json}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_reachable_tool_support_grpo_executable_smoke_v2}"
LOG="${LOG:-${STORE}/logs/muno21_reachable_tool_support_grpo_executable_smoke_v2.log}"
SFT_REPLAY="${SFT_REPLAY:-${STORE}/processed/muno21_v2/agent/reachable_tool_support_sft_v1/train/sft_composed.jsonl}"
POLL_SECONDS="${POLL_SECONDS:-30}"

cd "${PROJECT}"
mkdir -p "$(dirname "${OUTPUT}")" "$(dirname "${LOG}")"

if [[ -e "${OUTPUT}/summary.json" ]]; then
  echo "Refusing existing completed output: ${OUTPUT}/summary.json" >&2
  exit 1
fi

echo "Waiting for rollout diversity audit: ${AUDIT}"
until [[ -s "${AUDIT}" ]]; do
  sleep "${POLL_SECONDS}"
done

if ! grep -q '"ready_for_tool_belief_grpo": true' "${AUDIT}"; then
  echo "Diversity gate failed; executable GRPO was not started." >&2
  exit 3
fi

[[ -s "${SFT_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "Missing SFT adapter: ${SFT_ADAPTER}" >&2
  exit 4
}

rollout_args=()
writeback_args=()
for index in 0 1 2 3; do
  seed=$((20261121 + index))
  run="${ROOT}/rollout${index}_seed${seed}"
  trajectories="${run}/qwen3_4b_sft_tool_to_belief.jsonl"
  calls="${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  writeback="${run}/writeback/writeback.jsonl"
  echo "Waiting for executable writeback: ${writeback}"
  until [[ -s "${writeback}" ]]; do
    sleep "${POLL_SECONDS}"
  done
  for path in "${trajectories}" "${calls}" "${writeback}"; do
    [[ -s "${path}" ]] || { echo "Missing rollout artifact: ${path}" >&2; exit 5; }
  done
  rollout_args+=(--rollout "${trajectories}" "${calls}")
  writeback_args+=(--writeback "${writeback}")
done

export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

CUDA_VISIBLE_DEVICES="${GPU:-1}" "${PY}" scripts/train_recurrent_proxy_grpo.py \
  "${SFT_ADAPTER}" "${OUTPUT}" \
  "${rollout_args[@]}" "${writeback_args[@]}" \
  --reward-mode executable \
  --objective sequence-clip \
  --dynamic-sampling variable-only \
  --max-groups 64 \
  --false-edit-lagrange 0.25 \
  --false-edit-target 0.0966 \
  --false-edit-dual-lr 1.0 \
  --learning-rate 2.5e-7 \
  --entropy-coef 0.001 \
  --sequence-kl-coef 0.01 \
  --sft-replay "${SFT_REPLAY}" \
  --sft-replay-weight 0.10 \
  --sft-replay-probability 1.0 \
  --sft-tool-replay-fraction 0.50 \
  --gradient-accumulation 2 \
  --epochs 1 \
  --seed 20261220 \
  >"${LOG}" 2>&1

echo "Executable GRPO smoke completed: ${OUTPUT}"
