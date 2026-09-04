#!/usr/bin/env bash
set -euo pipefail

# One train-only diagnostic rollout for the prespecified safety-heavy SFT.
# It measures action support only and cannot be reported as a promoted result.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-4B}"
SFT="${SFT:-${STORE}/runs/agent/muno21_post_acquisition_safetyheavy_sft_20260807/final}"
DATA="${DATA:-${STORE}/runs/agent/muno21_post_acquisition_mixed_sft_v1_safetyheavy_20260807}"
STATES="${STATES:-${STORE}/processed/muno21_v2/agent/selector_states_v1.jsonl}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
SELECTOR="${SELECTOR:-${STORE}/runs/selector}"
BELIEF="${BELIEF:-${STORE}/runs/agent/muno21_post_acquisition_reliability_seed20260821/best_promoted.pt}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_post_acquisition_safetyheavy_action_screen_20260807}"
LOG="${LOG:-${STORE}/logs/muno21_post_acquisition_safetyheavy_action_screen_20260807.log}"

[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing output: ${OUTPUT}" >&2; exit 2; }
for path in "${PY}" "${MODEL}/config.json" "${SFT}/adapter_model.safetensors" \
  "${DATA}/train.jsonl" "${STATES}" "${EPISODES}" "${BELIEF}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done
mkdir -p "$(dirname "${LOG}")"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

CUDA_VISIBLE_DEVICES=7 "${PY}" scripts/evaluate_agent_rollouts.py \
  "${MODEL}" "${STATES}" "${OUTPUT}" --adapter "${SFT}" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
  --checkpoint "${SELECTOR}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
  --episodes "${EPISODES}" --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
  --tool-belief-checkpoint "${BELIEF}" --tool-supervision-jsonl "${DATA}/train.jsonl" \
  --max-tool-calls 2 --tool-out-size 256 --split train --oracle-step 1 \
  --budgets 1.5,3.0,4.5 --device cuda:0 --selector-device cpu \
  --seed 20260902 --sample-order seeded-hash --sample-seed 20260882 --ensure-tool-positive \
  --limit 264 --methods qwen3_4b_sft_tool_to_belief >"${LOG}" 2>&1
