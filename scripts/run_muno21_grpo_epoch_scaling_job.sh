#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed20260822/checkpoints/checkpoint-808}"
ROLLOUT_ROOT="${ROLLOUT_ROOT:-${STORAGE_ROOT}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n512_v1}"
STATES="${STATES:-${STORAGE_ROOT}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
BELIEF="${BELIEF:-${STORAGE_ROOT}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
SELECTOR_ROOT="${SELECTOR_ROOT:-${STORAGE_ROOT}/runs/selector}"
GPU="${GPU:?GPU is required}"
EPOCHS="${EPOCHS:?EPOCHS is required}"
SEED="${SEED:?SEED is required}"
SERVER_LABEL="${SERVER_LABEL:-unknown}"
ASSET_ROOT_MAP="${ASSET_ROOT_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/agent/muno21_grpo_epoch_scaling_v1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/muno21_grpo_epoch_scaling_v1}"
RUN="${RUN_ROOT}/epochs${EPOCHS}_seed${SEED}"
LOG="${LOG_ROOT}/epochs${EPOCHS}_seed${SEED}.log"

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
[[ ! -e "${RUN}" ]] || { echo "refusing existing output: ${RUN}" >&2; exit 4; }
for path in "${PYTHON}" "${MODEL}/config.json" "${SFT}/adapter_model.safetensors" \
  "${STATES}" "${EPISODES}" "${BELIEF}"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done

rollout_args=()
for index in 0 1 2 3 4 5 6 7; do
  rollout_seed=$((20260960 + index))
  rollout="${ROLLOUT_ROOT}/rollout${index}_seed${rollout_seed}"
  trajectories="${rollout}/qwen3_4b_sft_tool_to_belief.jsonl"
  calls="${rollout}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl"
  [[ -s "${trajectories}" && -s "${calls}" ]] || {
    echo "missing rollout pair: ${rollout}" >&2
    exit 3
  }
  rollout_args+=(--rollout "${trajectories}" "${calls}")
done

mkdir "${RUN}"
cat >"${RUN}/protocol.json" <<EOF
{
  "schema_version": "muno21-grpo-epoch-scaling-v1",
  "server": "${SERVER_LABEL}",
  "physical_gpu": ${GPU},
  "epochs": ${EPOCHS},
  "seed": ${SEED},
  "lambda_false_edit": 0.25,
  "false_edit_target": 0.0966,
  "learning_rate": 2.5e-7,
  "rollout_group_size": 8,
  "rollout_support": 512,
  "validation_tasks": 414,
  "split": "validation",
  "test_assets_read": false
}
EOF

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/train_recurrent_proxy_grpo.py \
  "${SFT}" "${RUN}/policy" "${rollout_args[@]}" \
  --objective sequence-clip --dynamic-sampling variable-only \
  --false-edit-lagrange 0.25 --false-edit-target 0.0966 \
  --false-edit-dual-lr 1.0 --minimum-keep-trajectories 100 \
  --minimum-false-edit-trajectories 1 --learning-rate 2.5e-7 \
  --entropy-coef 0.001 --gradient-accumulation 4 --epochs "${EPOCHS}" \
  --seed "${SEED}" >"${LOG}" 2>&1
train_code=$?
set -e
if [[ "${train_code}" -ne 0 ]]; then
  printf '{"stage":"train","exit_code":%s,"test_assets_read":false}\n' "${train_code}" >"${RUN}/FAILED.json"
  exit "${train_code}"
fi

OUTPUT="${RUN}/validation_full_gpu_selector"
set +e
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_rollouts.py \
  "${MODEL}" "${STATES}" "${OUTPUT}" --adapter "${RUN}/policy/final" \
  --device cuda:0 --selector-device cuda:0 \
  --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
  --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
  --checkpoint "${SELECTOR_ROOT}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
  --episodes "${EPISODES}" --tool-belief-checkpoint "${BELIEF}" \
  --tool-artifact-root "${OUTPUT}/tool_artifacts" --tool-out-size 256 \
  --asset-root-map "${ASSET_ROOT_MAP}" \
  --methods qwen3_4b_sft_tool_to_belief --split val --budgets 1.5,3.0,4.5 \
  --limit 512 --max-length 2048 --max-tool-calls 2 --seed "${SEED}" \
  >>"${LOG}" 2>&1
eval_code=$?
set -e
if [[ "${eval_code}" -ne 0 ]]; then
  printf '{"stage":"validation","exit_code":%s,"test_assets_read":false}\n' "${eval_code}" >"${RUN}/FAILED.json"
  exit "${eval_code}"
fi

printf '{"status":"complete","epochs":%s,"seed":%s,"server":"%s","test_assets_read":false}\n' \
  "${EPOCHS}" "${SEED}" "${SERVER_LABEL}" >"${RUN}/COMPLETED.json"
