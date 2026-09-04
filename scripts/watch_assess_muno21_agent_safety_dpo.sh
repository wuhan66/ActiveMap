#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_v2_persistent_seed20260821}"
SELECTION="$RUN_ROOT/evaluation/selection"
OUTPUT="$SELECTION/paired_promotion_decision.json"

required=(
  "$SELECTION/paired_action_bootstrap.json"
  "$SELECTION/paired_rollout_bootstrap.json"
  "$SELECTION/promotion_decision.json"
)
while true; do
  ready=true
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || ready=false
  done
  [[ "$ready" == "true" ]] && break
  sleep 60
done

cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  /home/wh/venvs/activemap/bin/python scripts/assess_agent_dpo_paired.py \
  "$SELECTION/paired_action_bootstrap.json" \
  "$SELECTION/paired_rollout_bootstrap.json" \
  "$SELECTION/promotion_decision.json" \
  "$OUTPUT"
