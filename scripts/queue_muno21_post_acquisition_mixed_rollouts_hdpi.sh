#!/usr/bin/env bash
set -euo pipefail

# Follow-on stage for the mixed cold start. It intentionally avoids GPU 2,
# which is occupied by a different project, and stops before writeback/GRPO
# if fresh no-override rollout coverage is inadequate.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_post_acquisition_mixed_reentry_v1_20260807}"
DATA="${DATA:-${STORE}/runs/agent/muno21_post_acquisition_mixed_sft_v1_data_20260807_r1}"
SFT_RUN="${SFT_RUN:-${ROOT}/sft_seed20260897}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_post_acquisition_mixed_reentry_v1_20260807}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
OBSOLETE_QUEUE_PID="${OBSOLETE_QUEUE_PID:-2719333}"

for path in "${PY}" "${MODEL}/config.json" "${DATA}/train.jsonl" "${STATES}" \
  "${EPISODES}" "${BELIEF}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done
[[ ! -e "${ROOT}/train_rollouts_g4" ]] || {
  echo "Refusing existing rollout root: ${ROOT}/train_rollouts_g4" >&2
  exit 2
}

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

while pgrep -f "train_agent_sft.py.*${SFT_RUN}" >/dev/null; do
  sleep 20
done
[[ -s "${SFT_RUN}/final/adapter_model.safetensors" ]] || {
  echo "SFT final adapter missing" >&2
  exit 4
}

# Retire only the suspended predecessor after its child trainer has exited.
if ps -p "${OBSOLETE_QUEUE_PID}" -o args= 2>/dev/null | grep -Fq "queue_muno21_post_acquisition_mixed_reentry_hdpi.sh"; then
  kill -TERM "${OBSOLETE_QUEUE_PID}"
fi

ROLLOUT_ROOT="${ROOT}/train_rollouts_g4"
mkdir -p "${ROLLOUT_ROOT}"
date -Is >"${ROOT}/ROLLOUT_QUEUE_STARTED"

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
launch_rollout 3 1 20260899 & pid1=$!
launch_rollout 4 2 20260900 & pid2=$!
launch_rollout 5 3 20260901 & pid3=$!
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
