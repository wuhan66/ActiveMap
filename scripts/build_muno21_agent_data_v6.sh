#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:?usage: build_muno21_agent_data_v6.sh SPLIT GPU_ID}"
GPU_ID="${2:?usage: build_muno21_agent_data_v6.sh SPLIT GPU_ID}"
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "val" ]]; then
  echo "split must be train or val" >&2
  exit 2
fi

PROJECT_ROOT="/home/wh/projects/activemap-v1"
PYTHON="/home/wh/venvs/activemap/bin/python"
DATA_ROOT="/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent"
OUTPUT_ROOT="${DATA_ROOT}/agent_data_v6_anonymized"
CHECKPOINT_ROOT="/mnt/mydisk/wh/ActiveMap/runs/selector"
CHECKPOINTS="${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260811/best.pt,${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260812/best.pt,${CHECKPOINT_ROOT}/muno21_evidence_conservative_v5_seed20260813/best.pt"

cd "${PROJECT_ROOT}"
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

"${PYTHON}" -m activemap.cli build-agent-data \
  "${DATA_ROOT}/selector_states_v1.jsonl" \
  "${OUTPUT_ROOT}/${SPLIT}" \
  --split "${SPLIT}" \
  --selector-checkpoints "${CHECKPOINTS}" \
  --device cuda \
  --top-k 3

"${PYTHON}" scripts/audit_agent_identifier_leakage.py \
  "${OUTPUT_ROOT}/${SPLIT}/sft.jsonl"

if [[ "${SPLIT}" == "train" ]]; then
  "${PYTHON}" scripts/balance_agent_sft.py \
    "${OUTPUT_ROOT}/train/sft.jsonl" \
    "${OUTPUT_ROOT}/train/sft_balanced.jsonl" \
    --acquire-repeat 3
  "${PYTHON}" scripts/audit_agent_identifier_leakage.py \
    "${OUTPUT_ROOT}/train/sft_balanced.jsonl"
fi
