#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1-joint-debug}"
STORE="${STORE:-/mnt/mydisk/wh/ActiveMap}"
PY="${PY:-/home/wh/venvs/activemap/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808}"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
ROOT="${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n1024_supportrep_v4"
MATRIX="${STORE}/runs/agent/muno21_grpo_n1024_supportrep_matrix_v4"
LOGS="${STORE}/logs/muno21_grpo_n1024_supportrep_matrix_v4"

mkdir -p "${ROOT}" "${LOGS}"
[[ ! -e "${MATRIX}" ]] || exit 0
for path in "${PY}" "${MODEL}/config.json" "${SFT}/adapter_model.safetensors" \
  "${STATES}" "${EPISODES}" "${BELIEF}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done

wait_for_capacity() {
  local gpu="$1" threshold="$2" used
  while true; do
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]')"
    [[ "${used}" =~ ^[0-9]+$ ]] || exit 3
    if (( used <= threshold )); then return 0; fi
    sleep 60
  done
}

# GPU0 retains an unrelated approximately 640 MiB process; wait only for our
# current job to release its memory. Other GPUs must be effectively empty.
mkdir -p "${MATRIX}"
cat >"${MATRIX}/protocol.json" <<EOF
{
  "schema_version": "muno21-grpo-n1024-support-replication-v4",
  "sample_order": "seeded-hash",
  "sample_seed": 20261040,
  "support_limit": 1024,
  "group_size": 8,
  "temperature": 2.2,
  "constrained_policy_seeds": [20261041, 20261042, 20261043],
  "unconstrained_diagnostic_seed": 20261044,
  "physical_gpus": [0, 1, 2, 3],
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
  local seed=$((20261040 + index))
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
    --asset-root-map "/home/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split train --budgets 1.5,3.0,4.5 \
    --limit 1024 --sample-order seeded-hash --sample-seed 20261040 \
    --max-length 2048 --max-tool-calls 2 --seed "${seed}" \
    --do-sample --temperature 2.2 --top-p 1.0 --record-training-payload \
    >"${LOGS}/rollout${index}_seed${seed}.log" 2>&1
}

(wait_for_capacity 0 1500; rollout_one 0 0; rollout_one 0 4) & p0=$!
(wait_for_capacity 1 256; rollout_one 1 1; rollout_one 1 5) & p1=$!
(wait_for_capacity 2 256; rollout_one 2 2; rollout_one 2 6) & p2=$!
(wait_for_capacity 3 256; rollout_one 3 3; rollout_one 3 7) & p3=$!
wait "${p0}" "${p1}" "${p2}" "${p3}"
date -Is >"${MATRIX}/ROLLOUTS_COMPLETED"

rollout_args=()
for index in 0 1 2 3 4 5 6 7; do
  seed=$((20261040 + index))
  run="${ROOT}/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

train_and_validate() {
  local gpu="$1" seed="$2" lambda="$3" label="$4"
  local run="${MATRIX}/${label}_seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${SFT}" "${run}/policy" "${rollout_args[@]}" \
    --objective sequence-clip --dynamic-sampling variable-only \
    --false-edit-lagrange "${lambda}" --false-edit-target 0.0966 \
    --false-edit-dual-lr 1.0 --minimum-keep-trajectories 100 \
    --minimum-false-edit-trajectories 1 --learning-rate 2.5e-7 \
    --entropy-coef 0.001 --gradient-accumulation 4 --epochs 1 --seed "${seed}" \
    >"${LOGS}/train_${label}_seed${seed}.log" 2>&1
  local output="${run}/validation_full_gpu_selector"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${run}/policy/final" \
    --device cuda:0 --selector-device cuda:0 \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
    --tool-artifact-root "${output}/tool_artifacts" --tool-out-size 256 \
    --asset-root-map "/home/wh/ActiveMap=${STORE}" \
    --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
    --limit 512 --max-length 2048 --max-tool-calls 2 --seed "${seed}" \
    >>"${LOGS}/train_${label}_seed${seed}.log" 2>&1
  printf '{"status":"complete","seed":%s,"lambda":%s,"test_assets_read":false}\n' \
    "${seed}" "${lambda}" >"${run}/COMPLETED.json"
}

(train_and_validate 0 20261041 0.25 lambda025) & t0=$!
(train_and_validate 1 20261042 0.25 lambda025) & t1=$!
(train_and_validate 2 20261043 0.25 lambda025) & t2=$!
(train_and_validate 3 20261044 0.0 unconstrained) & t3=$!
wait "${t0}" "${t1}" "${t2}" "${t3}"
date -Is >"${MATRIX}/MATRIX_COMPLETED"
