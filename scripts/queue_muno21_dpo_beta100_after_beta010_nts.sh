#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/mnt/mydisk/wh/ActiveMap}"
FIRST="${STORE}/runs/agent/muno21_qwen3_4b_safety_dpo_beta010_seed20260821_v1"
SECOND="${STORE}/runs/agent/muno21_qwen3_4b_safety_dpo_beta100_seed20260821_v1"

while [[ ! -s "${FIRST}/evaluation/selection/promotion_decision.json" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for beta010 formal decision"
  sleep 60
done

if [[ -s "${SECOND}/evaluation/selection/promotion_decision.json" ]]; then
  echo "beta100 formal evaluation already completed"
  exit 0
fi

cd "$PROJECT_ROOT"
MUNO21_DPO_RUN_DIR="$SECOND" MUNO21_AGENT_GPU=1 \
  bash scripts/watch_evaluate_muno21_agent_safety_dpo.sh
