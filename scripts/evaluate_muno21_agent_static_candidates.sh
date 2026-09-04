#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
PROJECT_ROOT="/home/wh/projects/activemap-v1"
PYTHON="/home/wh/venvs/activemap/bin/python"
OVERLAY="/mnt/mydisk/wh/ActiveMap/envs/agent_peft_overlay"
MODEL="/home/wh/hf_models/Qwen3-4B"
DATA_ROOT="/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/agent_data_v6_anonymized"
RUN_ROOT="/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821"

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${PROJECT_ROOT}/src:${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}"

for step in 400 418; do
  adapter="${RUN_ROOT}/checkpoints/checkpoint-${step}"
  output="${RUN_ROOT}/evaluation/checkpoint-${step}/actions"
  if [[ -s "${output}/summary.json" ]]; then
    echo "[$(date --iso-8601=seconds)] checkpoint-${step}: static result exists"
    continue
  fi
  echo "[$(date --iso-8601=seconds)] checkpoint-${step}: static action evaluation"
  "${PYTHON}" scripts/evaluate_agent_actions.py \
    "${MODEL}" "${DATA_ROOT}/val/sft.jsonl" "${output}" \
    --adapter "${adapter}" --device cuda --batch-size 1 \
    --max-length 2048 --max-new-tokens 64 \
    --trajectories-jsonl "${DATA_ROOT}/val/trajectories.jsonl"
  "${PYTHON}" scripts/analyze_agent_action_errors.py \
    "${output}/predictions.jsonl" "${output}/error_analysis.json"
done
