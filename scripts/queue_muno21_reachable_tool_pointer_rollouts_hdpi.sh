#!/usr/bin/env bash
set -euo pipefail

# Collect four aligned on-policy groups only after the pointer-action SFT
# adapter is complete. The diversity audit remains a hard gate for GRPO.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT_RUN="${SFT_RUN:-${STORE}/runs/agent/muno21_qwen3_4b_reachable_tool_pointer_sft_v3_seed20260842}"
SFT_ADAPTER="${SFT_ADAPTER:-${SFT_RUN}/final}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
TOOL_SUPERVISION="${TOOL_SUPERVISION:-${STORE}/runs/agent/reachable_tool_pointer_sft_v3/train.jsonl}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_rollouts_v3}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_reachable_tool_pointer_rollouts_v3}"

[[ ! -e "${ROOT}" ]] || { echo "Refusing existing rollout root: ${ROOT}" >&2; exit 2; }
for path in "${PY}" "${MODEL}/config.json" "${STATES}" "${EPISODES}" "${BELIEF}" "${TOOL_SUPERVISION}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done
mkdir -p "${ROOT}" "${LOGROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

until [[ -s "${SFT_ADAPTER}/adapter_model.safetensors" ]]; do sleep 60; done

# Saving ``final`` can precede teardown of the SFT CUDA context. Starting the
# rollout assigned to GPU 1 during that narrow window causes an avoidable OOM.
# Wait for the matching trainer to exit before reusing its device.
while pgrep -f "train_agent_sft.py.*${SFT_RUN}" >/dev/null; do
  echo "Waiting for pointer SFT CUDA teardown before rollout launch"
  sleep 30
done

launch_one() {
  local gpu="$1" index="$2" seed="$3"
  local output="${ROOT}/rollout${index}_seed${seed}"
  local log="${LOGROOT}/rollout${index}_seed${seed}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PY}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${output}" --adapter "${SFT_ADAPTER}" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --episodes "${EPISODES}" --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --tool-belief-checkpoint "${BELIEF}" --tool-supervision-jsonl "${TOOL_SUPERVISION}" \
    --max-tool-calls 2 --tool-out-size 256 --split train --oracle-step 1 \
    --budgets 1.5,3.0,4.5 --device cuda:0 --selector-device cpu --seed "${seed}" \
    --sample-order seeded-hash --sample-seed 20260882 --ensure-tool-positive --limit 264 \
    --do-sample --temperature 1.2 --top-p 0.95 --record-training-payload \
    --methods qwen3_4b_sft_tool_to_belief >"${log}" 2>&1 &
  echo "$! GPU${gpu} rollout${index} seed${seed}"
}

launch_one 1 0 20261251
launch_one 2 1 20261252
launch_one 3 2 20261253
launch_one 4 3 20261254
wait

args=()
for index in 0 1 2 3; do
  seed=$((20261251 + index))
  run="${ROOT}/rollout${index}_seed${seed}"
  args+=(--rollout "${run}/qwen3_4b_sft_tool_to_belief.jsonl" "${run}/llm_calls_qwen3_4b_sft_tool_to_belief.jsonl")
done
"${PY}" scripts/audit_recurrent_rollout_diversity.py "${ROOT}/diversity_audit.json" \
  "${args[@]}" --minimum-variable-group-rate 0.20 --minimum-nonstop-rate 0.05 \
  >"${LOGROOT}/diversity_audit.log" 2>&1
