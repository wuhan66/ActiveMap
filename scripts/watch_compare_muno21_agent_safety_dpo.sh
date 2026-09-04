#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_seed20260821}"
LABEL="safety-dpo-best"
BASE_EVALUATION="/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821/evaluation/checkpoint-200"
STATUS_PATH="${RUN_ROOT}/evaluation/selection/paired_comparison.exit_code"

mkdir -p "${RUN_ROOT}/evaluation/selection"
rm -f "$STATUS_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS_PATH"' EXIT

required=(
  "${BASE_EVALUATION}/actions/predictions.jsonl"
  "${BASE_EVALUATION}/rollouts/qwen3_4b_sft.jsonl"
  "${RUN_ROOT}/evaluation/${LABEL}/actions/predictions.jsonl"
  "${RUN_ROOT}/evaluation/${LABEL}/rollouts/qwen3_4b_sft.jsonl"
  "${RUN_ROOT}/evaluation/selection/promotion_decision.json"
)
while true; do
  ready=true
  for path in "${required[@]}"; do
    if [[ ! -s "$path" ]]; then
      ready=false
      break
    fi
  done
  if [[ "$ready" == "true" ]]; then
    break
  fi
  echo "[$(date --iso-8601=seconds)] waiting for formal DPO evaluation outputs"
  sleep 60
done

cd "$PROJECT_ROOT"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  /home/wh/venvs/activemap/bin/python scripts/compare_agent_action_predictions.py \
  "${BASE_EVALUATION}/actions/predictions.jsonl" \
  "${RUN_ROOT}/evaluation/${LABEL}/actions/predictions.jsonl" \
  "${RUN_ROOT}/evaluation/selection/paired_action_bootstrap.json" \
  --bootstrap 2000 --seed 20260821
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  /home/wh/venvs/activemap/bin/python scripts/compare_agent_rollouts.py \
  "${BASE_EVALUATION}/rollouts/qwen3_4b_sft.jsonl" \
  "${RUN_ROOT}/evaluation/${LABEL}/rollouts/qwen3_4b_sft.jsonl" \
  "${RUN_ROOT}/evaluation/selection/paired_rollout_bootstrap.json" \
  --bootstrap 2000 --seed 20260821

echo "[$(date --iso-8601=seconds)] formal safety DPO paired comparisons completed"
