#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT_RUN="${SFT_RUN:-${STORE}/runs/agent/muno21_qwen3_4b_reachable_tool_support_sft_seed20260831}"
SFT_ADAPTER="${SFT_ADAPTER:-${SFT_RUN}/final}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
TOOL_SUPERVISION="${TOOL_SUPERVISION:-${STORE}/runs/agent/reachable_tool_sft_v1/train.jsonl}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_rollouts_v1}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_reachable_tool_support_rollouts_v1}"

if [[ -e "${ROOT}/diversity_audit.json" ]]; then
  echo "Refusing existing audit: ${ROOT}/diversity_audit.json" >&2
  exit 1
fi
for path in "${PY}" "${MODEL}/config.json" "${STATES}" "${EPISODES}" \
  "${BELIEF}" "${TOOL_SUPERVISION}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

mkdir -p "${ROOT}" "${LOGROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

until [[ -s "${SFT_ADAPTER}/adapter_model.safetensors" ]]; do
  sleep 60
done

launch_one() {
  local gpu="$1"
  local index="$2"
  local seed="$3"
  local output="${ROOT}/rollout${index}_seed${seed}"
  local log="${LOGROOT}/rollout${index}_seed${seed}.log"
  if [[ -e "${output}" ]]; then
    echo "Refusing existing output: ${output}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" \
    --adapter "${SFT_ADAPTER}" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --tool-belief-checkpoint "${BELIEF}" \
    --tool-supervision-jsonl "${TOOL_SUPERVISION}" \
    --max-tool-calls 2 --tool-out-size 256 \
    --split train --oracle-step 1 --budgets 1.5,3.0,4.5 \
    --device cuda:0 --selector-device cpu \
    --seed "${seed}" --sample-order seeded-hash --sample-seed 20260882 \
    --ensure-tool-positive --limit 264 --do-sample \
    --temperature 2.2 --top-p 1.0 --record-training-payload \
    --methods qwen3_4b_sft_tool_to_belief \
    >"${log}" 2>&1 &
  echo "$! GPU${gpu} rollout${index} seed${seed}"
}

launch_one 1 0 20261121
launch_one 2 1 20261122
launch_one 3 2 20261123
launch_one 4 3 20261124
wait

rollout_args=()
for index in 0 1 2 3; do
  case "${index}" in
    0) seed=20261121 ;;
    1) seed=20261122 ;;
    2) seed=20261123 ;;
    3) seed=20261124 ;;
  esac
  run="${ROOT}/rollout${index}_seed${seed}"
  rollout_args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done

"${PY}" scripts/audit_recurrent_rollout_diversity.py \
  "${ROOT}/diversity_audit.json" "${rollout_args[@]}" \
  --minimum-variable-group-rate 0.20 --minimum-nonstop-rate 0.05 \
  >"${LOGROOT}/diversity_audit.log" 2>&1
