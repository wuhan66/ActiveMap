#!/usr/bin/env bash
set -euo pipefail

# Greedy validation is kept separate from sampled train collection. It reports
# whether the learned pointer interface actually reaches executable tools.
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
TOOL_SUPERVISION="${TOOL_SUPERVISION:-${STORE}/runs/agent/reachable_tool_pointer_sft_v3/val.jsonl}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_validation_v3}"
LOG="${LOG:-${STORE}/logs/muno21_reachable_tool_pointer_validation_v3.log}"
GPU="${GPU:-5}"

[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing output: ${OUTPUT}" >&2; exit 2; }
for path in "${PY}" "${MODEL}/config.json" "${STATES}" "${EPISODES}" "${BELIEF}" "${TOOL_SUPERVISION}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done
until [[ -s "${SFT_ADAPTER}/adapter_model.safetensors" ]]; do sleep 60; done

cd "${PROJECT}"
mkdir -p "$(dirname "${LOG}")"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/evaluate_agent_rollouts.py \
  "${MODEL}" "${STATES}" "${OUTPUT}" --adapter "${SFT_ADAPTER}" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
  --episodes "${EPISODES}" --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
  --tool-belief-checkpoint "${BELIEF}" --tool-artifact-root "${OUTPUT}/tool_artifacts" \
  --tool-supervision-jsonl "${TOOL_SUPERVISION}" --max-tool-calls 2 --tool-out-size 256 \
  --split val --oracle-step 1 --budgets 1.5,3.0,4.5 --device cuda:0 --selector-device cpu \
  --seed 20260842 --sample-order seeded-hash --sample-seed 20260882 --ensure-tool-positive \
  --limit 22 --methods qwen3_4b_sft_tool_to_belief >"${LOG}" 2>&1

"${PY}" scripts/audit_pointer_action_validation.py \
  "${OUTPUT}/summary.json" "${OUTPUT}/pointer_action_audit.json" \
  >>"${LOG}" 2>&1

echo "Pointer-action greedy validation completed: ${OUTPUT}"
