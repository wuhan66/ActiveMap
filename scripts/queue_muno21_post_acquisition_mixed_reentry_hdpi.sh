#!/usr/bin/env bash
set -euo pipefail

# A bounded re-entry branch. It trains one balanced cold start, then collects
# four fresh no-override train rollouts. Any failed audit stops before map
# writeback or GRPO; this is deliberately not a multi-seed promotion sweep.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
DATA="${DATA:-${STORE}/runs/agent/muno21_post_acquisition_mixed_sft_v1_data_20260807_r1}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_post_acquisition_mixed_reentry_v1_20260807}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"

[[ ! -e "${ROOT}" ]] || { echo "Refusing existing run root: ${ROOT}" >&2; exit 2; }
for path in "${PY}" "${MODEL}/config.json" "${DATA}/train.jsonl" "${DATA}/val.jsonl" \
  "${DATA}/action_sft_audit.json" "${STATES}" "${EPISODES}" "${BELIEF}" "${UPDATER}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done

mkdir -p "${ROOT}" "${LOGROOT}"
cat >"${ROOT}/protocol.json" <<EOF
{
  "schema_version": "muno21-post-acquisition-mixed-reentry-v1",
  "purpose": "repair all-tool/all-COMMIT action-support collapse before recurrent GRPO",
  "cold_start_data": "${DATA}",
  "train_action_balance": ["REJECT", "COMMIT", "USE_TOOL"],
  "validation_selection": "last checkpoint; validation loss is diagnostic only",
  "rollouts": 4,
  "rollout_initial_state_sampling": "seeded-hash-20260882",
  "behavior_overrides": false,
  "promotion": false,
  "test_assets_read": false
}
EOF

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

SFT_RUN="${ROOT}/sft_seed20260897"
CUDA_VISIBLE_DEVICES=1 "${PY}" scripts/train_agent_sft.py \
  "${MODEL}" "${DATA}/train.jsonl" "${SFT_RUN}" \
  --eval-jsonl "${DATA}/val.jsonl" --epochs 3 --learning-rate 0.0001 \
  --batch-size 1 --gradient-accumulation 16 --max-length 2048 \
  --logging-steps 10 --eval-steps 30 --save-steps 30 --save-total-limit 3 \
  --select-last-checkpoint --seed 20260897 >"${LOGROOT}/sft.log" 2>&1
[[ -s "${SFT_RUN}/final/adapter_model.safetensors" ]] || {
  echo "SFT final adapter missing" >&2
  exit 4
}

ROLLOUT_ROOT="${ROOT}/train_rollouts_g4"
mkdir -p "${ROLLOUT_ROOT}"
launch_rollout() {
  local gpu="$1" index="$2" seed="$3"
  local output="${ROLLOUT_ROOT}/rollout${index}_seed${seed}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${SFT_RUN}/final" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --tool-belief-checkpoint "${BELIEF}" --tool-supervision-jsonl "${DATA}/train.jsonl" \
    --max-tool-calls 2 --tool-out-size 256 --split train --oracle-step 1 \
    --budgets 1.5,3.0,4.5 --device cuda:0 --selector-device cpu \
    --seed "${seed}" --sample-order seeded-hash --sample-seed 20260882 \
    --ensure-tool-positive --limit 264 --do-sample --temperature 1.0 --top-p 0.95 \
    --record-training-payload --methods qwen3_4b_sft_tool_to_belief \
    >"${LOGROOT}/rollout${index}_seed${seed}.log" 2>&1
}

launch_rollout 1 0 20260898 & pid0=$!
launch_rollout 2 1 20260899 & pid1=$!
launch_rollout 3 2 20260900 & pid2=$!
launch_rollout 4 3 20260901 & pid3=$!
wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"

rollout_args=()
for index in 0 1 2 3; do
  seed=$((20260898 + index))
  run="${ROLLOUT_ROOT}/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" \
    "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

if ! "${PY}" scripts/audit_recurrent_rollout_diversity.py \
  "${ROOT}/diversity_audit.json" "${rollout_args[@]}" \
  --minimum-variable-group-rate 0.20 --minimum-nonstop-rate 0.05 \
  >"${LOGROOT}/diversity_audit.log" 2>&1; then
  date -Is >"${ROOT}/NOT_READY_DIVERSITY"
  exit 0
fi
if ! "${PY}" scripts/audit_recurrent_rollout_coverage.py \
  "${ROOT}/executed_coverage_audit.json" "${rollout_args[@]}" \
  --minimum-keep-trajectories 16 --minimum-commit-trajectories 16 \
  --minimum-tool-trajectories 16 --maximum-fallback-rate 0.01 \
  >"${LOGROOT}/executed_coverage_audit.log" 2>&1; then
  date -Is >"${ROOT}/NOT_READY_EXECUTED_COVERAGE"
  exit 0
fi

date -Is >"${ROOT}/READY_FOR_EXECUTABLE_WRITEBACK"
