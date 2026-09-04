#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="${MUNO21_8B_MODEL:-/home/wh/hf_models/Qwen3-8B}"
DATA="${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools"
OUTPUT="${STORAGE_ROOT}/runs/smoke/muno21_qwen3_8b_balanced_tool_sft_2sample_v1"
GPU="${GPU:-5}"

case "${GPU}" in
  1|2|3|4|5|7) ;;
  0|6) echo "GPU${GPU} is reserved" >&2; exit 2 ;;
  *) echo "unsupported physical GPU id: ${GPU}" >&2; exit 2 ;;
esac
for path in \
  "${MODEL}/config.json" \
  "${DATA}/train/sft_composed.jsonl" \
  "${DATA}/val/sft_composed.jsonl"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done
[[ ! -e "${OUTPUT}" ]] || {
  [[ -s "${OUTPUT}/final/adapter_config.json" ]] && exit 0
  echo "refusing partial smoke output: ${OUTPUT}" >&2
  exit 4
}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/train_agent_sft.py \
  "${MODEL}" "${DATA}/train/sft_composed.jsonl" "${OUTPUT}" \
  --eval-jsonl "${DATA}/val/sft_composed.jsonl" \
  --epochs 1 --learning-rate 0.0002 --batch-size 1 \
  --gradient-accumulation 1 --max-length 2048 \
  --logging-steps 1 --eval-steps 1 --save-steps 1 --save-total-limit 1 \
  --early-stopping-patience 0 \
  --lora-rank 16 --lora-alpha 32 --seed 20260908 \
  --max-train-samples 2 --max-eval-samples 2
