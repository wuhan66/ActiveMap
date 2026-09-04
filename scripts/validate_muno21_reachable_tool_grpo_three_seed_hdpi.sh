#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_qwen3_4b_reachable_tool_support_sft_seed20260831/final}"
GRPO_ROOT="${GRPO_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_grpo_three_seed_v1}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
TOOL_SUPERVISION="${TOOL_SUPERVISION:-${STORE}/runs/agent/reachable_tool_sft_v1/val.jsonl}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_grpo_three_seed_validation_v1}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_reachable_tool_support_grpo_three_seed_validation_v1}"

for path in "${PY}" "${MODEL}/config.json" "${SFT}/adapter_model.safetensors" \
  "${STATES}" "${EPISODES}" "${BELIEF}" "${TOOL_SUPERVISION}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
for seed in 20261220 20261221 20261222; do
  [[ -s "${GRPO_ROOT}/seed${seed}/final/adapter_model.safetensors" ]] || {
    echo "Missing GRPO adapter for seed ${seed}" >&2
    exit 3
  }
done

mkdir -p "${OUTPUT_ROOT}" "${LOGROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

validate_one() {
  local gpu="$1"
  local name="$2"
  local adapter="$3"
  local seed="$4"
  local output="${OUTPUT_ROOT}/${name}_seed${seed}"
  local log="${LOGROOT}/${name}_seed${seed}.log"
  if [[ -s "${output}/summary.json" ]]; then
    echo "validation already complete: ${output}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
    nohup "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${adapter}" \
    --device cuda:0 --selector-device cpu \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --tool-supervision-jsonl "${TOOL_SUPERVISION}" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --oracle-step 1 \
    --budgets 1.5,3.0,4.5 --limit 33 --max-length 2048 --max-tool-calls 2 \
    --seed "${seed}" --sample-order seeded-hash --sample-seed 20260882 \
    --ensure-tool-positive --do-sample --temperature 2.2 --top-p 1.0 \
    --record-training-payload \
    >"${log}" 2>&1 &
  echo "$! GPU${gpu} ${name} seed${seed}"
}

validate_one 1 candidate "${GRPO_ROOT}/seed20261220/final" 20261220
validate_one 2 candidate "${GRPO_ROOT}/seed20261221/final" 20261221
validate_one 3 candidate "${GRPO_ROOT}/seed20261222/final" 20261222
validate_one 4 sft_control "${SFT}" 20261220
validate_one 5 sft_control "${SFT}" 20261221
validate_one 7 sft_control "${SFT}" 20261222
wait
echo "three-seed GRPO and matched SFT validation completed"
