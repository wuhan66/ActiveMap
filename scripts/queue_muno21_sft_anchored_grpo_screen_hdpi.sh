#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808}"
SFT_REPLAY="${SFT_REPLAY:-${STORE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/train/sft_composed.jsonl}"
STATES="${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl"
EPISODES="${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
BELIEF="${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt"
SELECTOR="${STORE}/runs/selector"
ROLLOUT_ROOT="${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n1024_v1"
MATRIX="${STORE}/runs/agent/muno21_sft_anchored_grpo_screen_v1"
LOGS="${STORE}/logs/muno21_sft_anchored_grpo_screen_v1"

[[ ! -e "${MATRIX}" ]] || exit 0
for path in "${PY}" "${MODEL}/config.json" "${SFT}/adapter_model.safetensors" \
  "${SFT_REPLAY}" "${STATES}" "${EPISODES}" "${BELIEF}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
mkdir -p "${MATRIX}" "${LOGS}"

cat >"${MATRIX}/protocol.json" <<EOF
{
  "schema_version": "muno21-sft-anchored-constrained-grpo-screen-v1",
  "rollout_support": "seeded-hash-g8-n1024-20261020",
  "sft_replay_weights": [0.05, 0.10, 0.20],
  "sft_replay_probability": 0.25,
  "sft_tool_replay_fraction": 0.50,
  "sequence_kl_coef": 0.01,
  "false_edit_lagrange": 0.25,
  "shared_policy_seed": 20261201,
  "physical_gpus": [1, 2, 7],
  "role": "single-seed hyperparameter screen before three-seed replication",
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

rollout_args=()
for index in 0 1 2 3 4 5 6 7; do
  seed=$((20261020 + index))
  run="${ROLLOUT_ROOT}/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

wait_for_gpu() {
  local gpu="$1"
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    sleep 60
  done
}

train_and_validate() {
  local gpu="$1" weight="$2" label="$3"
  wait_for_gpu "${gpu}"
  local run="${MATRIX}/${label}_seed20261201"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/train_recurrent_proxy_grpo.py \
    "${SFT}" "${run}/policy" "${rollout_args[@]}" \
    --objective sequence-clip --dynamic-sampling variable-only \
    --false-edit-lagrange 0.25 --false-edit-target 0.0966 \
    --false-edit-dual-lr 1.0 --minimum-keep-trajectories 100 \
    --minimum-false-edit-trajectories 1 --learning-rate 2.5e-7 \
    --entropy-coef 0.001 --sequence-kl-coef 0.01 \
    --sft-replay "${SFT_REPLAY}" --sft-replay-weight "${weight}" \
    --sft-replay-probability 0.25 --sft-tool-replay-fraction 0.50 \
    --gradient-accumulation 4 \
    --epochs 1 --seed 20261201 >"${LOGS}/${label}.log" 2>&1

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
    --limit 512 --max-length 2048 --max-tool-calls 2 --seed 20261201 \
    >>"${LOGS}/${label}.log" 2>&1
  printf '{"status":"complete","weight":%s,"seed":20261201,"test_assets_read":false}\n' \
    "${weight}" >"${run}/COMPLETED.json"
}

(train_and_validate 1 0.05 sftw005) & p1=$!
(train_and_validate 2 0.10 sftw010) & p2=$!
(train_and_validate 7 0.20 sftw020) & p7=$!
wait "${p1}" "${p2}" "${p7}"
date -Is >"${MATRIX}/MATRIX_COMPLETED"
