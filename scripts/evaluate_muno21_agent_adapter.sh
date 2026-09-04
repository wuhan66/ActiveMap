#!/usr/bin/env bash
set -euo pipefail

ADAPTER="${1:?usage: evaluate_muno21_agent_adapter.sh ADAPTER LABEL [GPU]}"
LABEL="${2:?usage: evaluate_muno21_agent_adapter.sh ADAPTER LABEL [GPU]}"
GPU="${3:-2}"

PROJECT_ROOT="/home/wh/projects/activemap-v1"
PYTHON="/home/wh/venvs/activemap/bin/python"
OVERLAY="/mnt/mydisk/wh/ActiveMap/envs/agent_peft_overlay"
MODEL="/home/wh/hf_models/Qwen3-4B"
DATA_ROOT="/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent"
AGENT_DATA="${DATA_ROOT}/agent_data_v6_anonymized"
STATES="${DATA_ROOT}/selector_states_v1.jsonl"
RUN_ROOT="/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821"
OUTPUT_ROOT="${RUN_ROOT}/evaluation/${LABEL}"
CHECKPOINT_ROOT="/mnt/mydisk/wh/ActiveMap/runs/selector"

[[ -s "${ADAPTER}/adapter_model.safetensors" ]] || {
  echo "Missing adapter: ${ADAPTER}/adapter_model.safetensors" >&2
  exit 1
}
mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}/src:${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "${OUTPUT_ROOT}/actions/summary.json" ]]; then
  echo "[$(date --iso-8601=seconds)] ${LABEL}: static action evaluation"
  "${PYTHON}" scripts/evaluate_agent_actions.py \
    "${MODEL}" "${AGENT_DATA}/val/sft.jsonl" "${OUTPUT_ROOT}/actions" \
    --adapter "${ADAPTER}" --device cuda --batch-size 1 \
    --max-length 2048 --max-new-tokens 64 \
    --trajectories-jsonl "${AGENT_DATA}/val/trajectories.jsonl"
else
  echo "[$(date --iso-8601=seconds)] ${LABEL}: static result already exists"
fi

if [[ ! -s "${OUTPUT_ROOT}/actions/error_analysis.json" ]]; then
  "${PYTHON}" scripts/analyze_agent_action_errors.py \
    "${OUTPUT_ROOT}/actions/predictions.jsonl" \
    "${OUTPUT_ROOT}/actions/error_analysis.json"
fi

if [[ ! -s "${OUTPUT_ROOT}/rollouts/summary.json" ]]; then
  echo "[$(date --iso-8601=seconds)] ${LABEL}: closed-loop evaluation"
  "${PYTHON}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${STATES}" "${OUTPUT_ROOT}/rollouts" \
    --adapter "${ADAPTER}" --device cuda --selector-device cpu \
    --checkpoint "${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260811/best.pt" \
    --checkpoint "${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260812/best.pt" \
    --checkpoint "${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260813/best.pt" \
    --split val --budgets 1.5,3.0,4.5 --max-length 2048
else
  echo "[$(date --iso-8601=seconds)] ${LABEL}: rollout result already exists"
fi

echo "[$(date --iso-8601=seconds)] ${LABEL}: complete"
