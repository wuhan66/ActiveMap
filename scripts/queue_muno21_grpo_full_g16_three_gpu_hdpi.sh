#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808}"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
WAIT_FOR="${STORE}/runs/agent/muno21_grpo_n2048_lambda025_matrix_v1/MATRIX_COMPLETED"
ROOT="${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g16_full_v1"
MATRIX="${STORE}/runs/agent/muno21_grpo_full_g16_lambda025_matrix_v1"
LOGS="${STORE}/logs/muno21_grpo_full_g16_lambda025_matrix_v1"
SUPPORT_LIMIT="${SUPPORT_LIMIT:-2544}"

mkdir -p "${ROOT}" "${LOGS}"
[[ ! -e "${MATRIX}" ]] || exit 0
while [[ ! -s "${WAIT_FOR}" ]]; do sleep 60; done
for gpu in 3 4 5; do
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    sleep 60
  done
done

mkdir -p "${MATRIX}"
cat >"${MATRIX}/protocol.json" <<EOF
{
  "schema_version": "muno21-grpo-full-g16-lambda025-v1",
  "sample_order": "seeded-hash",
  "sample_seed": 20261100,
  "support_limit": ${SUPPORT_LIMIT},
  "group_size": 16,
  "temperature": 2.2,
  "lambda_false_edit": 0.25,
  "policy_seeds": [20261121, 20261122, 20261123],
  "physical_gpus": [3, 4, 5],
  "role": "full-support high-sampling stress endpoint",
  "split": "train_then_validation",
  "test_assets_read": false
}
EOF

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

rollout_one() {
  local gpu="$1" index="$2"
  local seed=$((20261100 + index))
  local output="${ROOT}/rollout${index}_seed${seed}"
  [[ ! -e "${output}" ]] || { echo "refusing existing rollout: ${output}" >&2; return 4; }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${SFT}" \
    --device cuda:0 --selector-device cuda:0 \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split train --budgets 1.5,3.0,4.5 \
    --limit "${SUPPORT_LIMIT}" --sample-order seeded-hash --sample-seed 20261100 \
    --max-length 2048 --max-tool-calls 2 --seed "${seed}" \
    --do-sample --temperature 2.2 --top-p 1.0 --record-training-payload \
    >"${LOGS}/rollout${index}_seed${seed}.log" 2>&1
}

(for i in 0 3 6 9 12 15; do rollout_one 3 "${i}"; done) & p3=$!
(for i in 1 4 7 10 13; do rollout_one 4 "${i}"; done) & p4=$!
(for i in 2 5 8 11 14; do rollout_one 5 "${i}"; done) & p5=$!
wait "${p3}" "${p4}" "${p5}"
date -Is >"${MATRIX}/ROLLOUTS_COMPLETED"

rollout_args=()
for index in $(seq 0 15); do
  seed=$((20261100 + index))
  run="${ROOT}/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

train_and_validate() {
  local gpu="$1" seed="$2"
  local run="${MATRIX}/lambda025_seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${SFT}" "${run}/policy" "${rollout_args[@]}" \
    --objective sequence-clip --dynamic-sampling variable-only \
    --false-edit-lagrange 0.25 --false-edit-target 0.0966 \
    --false-edit-dual-lr 1.0 --minimum-keep-trajectories 100 \
    --minimum-false-edit-trajectories 1 --learning-rate 2.5e-7 \
    --entropy-coef 0.001 --gradient-accumulation 4 --epochs 1 --seed "${seed}" \
    >"${LOGS}/train_seed${seed}.log" 2>&1
  local output="${run}/validation_full_gpu_selector"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${run}/policy/final" \
    --device cuda:0 --selector-device cuda:0 \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit 512 --max-length 2048 --max-tool-calls 2 --seed "${seed}" \
    >>"${LOGS}/train_seed${seed}.log" 2>&1
  printf '{"status":"complete","seed":%s,"test_assets_read":false}\n' "${seed}" \
    >"${run}/COMPLETED.json"
}

(train_and_validate 3 20261121) & t3=$!
(train_and_validate 4 20261122) & t4=$!
(train_and_validate 5 20261123) & t5=$!
wait "${t3}" "${t4}" "${t5}"
date -Is >"${MATRIX}/MATRIX_COMPLETED"
