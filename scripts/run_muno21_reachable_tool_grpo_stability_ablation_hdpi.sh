#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
SFT_ADAPTER="${SFT_ADAPTER:-${STORE}/runs/agent/muno21_qwen3_4b_reachable_tool_support_sft_seed20260831/final}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_rollouts_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_grpo_stability_ablation_v1}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_reachable_tool_support_grpo_stability_ablation_v1}"
SFT_REPLAY="${SFT_REPLAY:-${STORE}/processed/muno21_v2/agent/reachable_tool_support_sft_v1/train/sft_composed.jsonl}"

for path in "${PY}" "${SFT_ADAPTER}/adapter_model.safetensors" "${SFT_REPLAY}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_ROOT}" "${LOGROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

rollout_args=()
writeback_args=()
for index in 0 1 2 3; do
  seed=$((20261121 + index))
  run="${ROOT}/rollout${index}_seed${seed}"
  trajectories="${run}/qwen3_4b_sft_tool_to_belief.jsonl"
  calls="${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  writeback="${run}/writeback/writeback.jsonl"
  for path in "${trajectories}" "${calls}" "${writeback}"; do
    [[ -s "${path}" ]] || { echo "Missing rollout artifact: ${path}" >&2; exit 3; }
  done
  rollout_args+=(--rollout "${trajectories}" "${calls}")
  writeback_args+=(--writeback "${writeback}")
done

launch_one() {
  local gpu="$1"
  local name="$2"
  local seed="$3"
  shift 3
  local output="${OUTPUT_ROOT}/${name}"
  local log="${LOGROOT}/${name}.log"
  local extra=("$@")
  if [[ -s "${output}/COMPLETED" && -s "${output}/final/adapter_model.safetensors" ]]; then
    echo "GRPO ablation already complete: ${name}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${SFT_ADAPTER}" "${output}" \
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
    --sft-replay-probability 1.0 \
    --gradient-accumulation 2 \
    --epochs 1 \
    --seed "${seed}" \
    "${extra[@]}" \
    >"${log}" 2>&1 &
  echo "$! GPU${gpu} ${name}"
}

launch_one 1 sft_anchor05 20261230 \
  --sft-replay-weight 0.50 --sft-tool-replay-fraction 0.75 \
  --invalid-action-weight 1.0
launch_one 2 invalid_penalty2 20261231 \
  --sft-replay-weight 0.10 --sft-tool-replay-fraction 0.50 \
  --invalid-action-weight 2.0
wait

for name in sft_anchor05 invalid_penalty2; do
  [[ -s "${OUTPUT_ROOT}/${name}/COMPLETED" ]] || {
    echo "GRPO ablation did not complete: ${name}" >&2
    exit 4
  }
done

echo "GRPO stability ablations completed"
