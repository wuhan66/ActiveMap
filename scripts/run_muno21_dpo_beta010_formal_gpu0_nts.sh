#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
RUN="${RUN:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_beta010_seed20260821_v1}"
PYTHON="${PYTHON:-/home/wh/venvs/activemap/bin/python}"
LABEL="safety-dpo-best"

cd "$PROJECT"
if [[ ! -s "${RUN}/evaluation/selection/promotion_decision.json" ]]; then
  bash scripts/evaluate_muno21_agent_run_adapter.sh \
    "$RUN" "${RUN}/final" "$LABEL" 0
  PYTHONPATH="${PROJECT}/src${PYTHONPATH:+:${PYTHONPATH}}" "$PYTHON" \
    scripts/summarize_agent_checkpoints.py \
    "${RUN}/evaluation" "${RUN}/evaluation/selection" --labels "$LABEL"
fi

MUNO21_DPO_RUN_DIR="$RUN" bash scripts/watch_assess_muno21_agent_safety_dpo.sh
