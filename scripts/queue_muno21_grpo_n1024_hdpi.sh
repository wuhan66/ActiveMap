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
ROOT="${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n1024_v1"
MATRIX="${STORE}/runs/agent/muno21_grpo_n1024_lambda025_matrix_v1"
LOGS="${STORE}/logs/muno21_grpo_n1024_lambda025_matrix_v1"
GPUS=(1 2 3 4 5 7)

mkdir -p "${ROOT}" "${LOGS}"
[[ ! -e "${MATRIX}" ]] || exit 0
for path in "${PY}" "${MODEL}/config.json" "${SFT}/adapter_model.safetensors" \
  "${STATES}" "${EPISODES}" "${BELIEF}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done

wait_for_gpu() {
  local gpu="$1"
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    sleep 60
  done
}

mkdir -p "${MATRIX}"
cat >"${MATRIX}/protocol.json" <<EOF
{
  "schema_version": "muno21-grpo-n1024-lambda025-v1",
  "sample_order": "seeded-hash",
  "sample_seed": 20261020,
  "support_limit": 1024,
  "group_size": 8,
  "temperature": 2.2,
  "lambda_false_edit": 0.25,
  "training_epochs": 1,
  "policy_seeds": [20261021, 20261022, 20261023],
  "physical_gpus": [1, 2, 3, 4, 5, 7],
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
  local seed=$((20261020 + index))
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
    --limit 1024 --sample-order seeded-hash --sample-seed 20261020 \
    --max-length 2048 --max-tool-calls 2 --seed "${seed}" \
    --do-sample --temperature 2.2 --top-p 1.0 --record-training-payload \
    >"${LOGS}/rollout${index}_seed${seed}.log" 2>&1
}

(wait_for_gpu 1; rollout_one 1 0; rollout_one 1 6) & p1=$!
(wait_for_gpu 2; rollout_one 2 1; rollout_one 2 7) & p2=$!
(wait_for_gpu 3; rollout_one 3 2) & p3=$!
(wait_for_gpu 4; rollout_one 4 3) & p4=$!
(wait_for_gpu 5; rollout_one 5 4) & p5=$!
(wait_for_gpu 7; rollout_one 7 5) & p7=$!
wait "${p1}" "${p2}" "${p3}" "${p4}" "${p5}" "${p7}"
date -Is >"${MATRIX}/ROLLOUTS_COMPLETED"

rollout_args=()
for index in 0 1 2 3 4 5 6 7; do
  seed=$((20261020 + index))
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

(train_and_validate 3 20261021) & t1=$!
(train_and_validate 4 20261022) & t2=$!
(train_and_validate 5 20261023) & t3=$!
wait "${t1}" "${t2}" "${t3}"
date -Is >"${MATRIX}/MATRIX_COMPLETED"
