#!/usr/bin/env bash
set -euo pipefail

# Four-GPU replication is intentionally blocked on an explicit single-GPU
# smoke marker. It never trains GRPO or writes map edits: its sole purpose is
# to collect aligned, valid, diverse on-policy trajectories.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_candidate_decoder_reentry_v1_20260807}"
SMOKE_ROOT="${SMOKE_ROOT:-${STORE}/runs/agent/muno21_post_acquisition_candidate_decoder_smoke_20260807}"
SFT_RUN="${SFT_RUN:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807/sft_seed20260897}"
DATA="${DATA:-${STORE}/runs/agent/muno21_post_acquisition_mixed_sft_v1_data_20260807_r1}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_reentry_trainval_v1_20260807.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_candidate_decoder_reentry_v1_20260807}"

for path in "${PY}" "${MODEL}/config.json" "${SFT_RUN}/final/adapter_model.safetensors" \
  "${DATA}/train.jsonl" "${STATES}" "${EPISODES}" "${BELIEF}" \
  "${SMOKE_ROOT}/SMOKE_READY_FOR_FOUR_GPU"; do
  [[ -s "${path}" ]] || { echo "Missing required input: ${path}" >&2; exit 3; }
done
[[ ! -e "${ROOT}/candidate_rollouts_g4" ]] || {
  echo "Refusing existing output: ${ROOT}/candidate_rollouts_g4" >&2
  exit 2
}

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
mkdir -p "${ROOT}/candidate_rollouts_g4" "${LOGROOT}"
date -Is >"${ROOT}/QUEUE_STARTED"

launch_rollout() {
  local gpu="$1" index="$2" seed="$3"
  local output="${ROOT}/candidate_rollouts_g4/rollout${index}_seed${seed}"
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
    --ensure-tool-positive --limit 120 --do-sample --temperature 1.5 --top-p 0.95 \
    --action-decoder candidate-sample --candidate-score-batch-size 1 \
    --record-training-payload --methods qwen3_4b_sft_tool_to_belief \
    >"${LOGROOT}/rollout${index}_seed${seed}.log" 2>&1
}

# At most four GPUs by construction. GPU 0 and GPU 2 are never touched.
launch_rollout 1 0 20260910 & pid0=$!
launch_rollout 3 1 20260911 & pid1=$!
launch_rollout 4 2 20260912 & pid2=$!
launch_rollout 5 3 20260913 & pid3=$!
wait "${pid0}" "${pid1}" "${pid2}" "${pid3}"

rollout_args=()
for index in 0 1 2 3; do
  seed=$((20260910 + index))
  run="${ROOT}/candidate_rollouts_g4/rollout${index}_seed${seed}"
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
