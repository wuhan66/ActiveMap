#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_ROOT="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_v2_persistent_seed20260821}"
SELECTION="${RUN_ROOT}/evaluation/selection"
STATUS_PATH="${SELECTION}/paired_assessment_quality_v2.exit_code"
OUTPUT="${SELECTION}/paired_promotion_decision_quality_v2.json"

mkdir -p "$SELECTION"
rm -f "$STATUS_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS_PATH"' EXIT

required=(
  "${SELECTION}/paired_comparison_quality_v2.exit_code"
  "${SELECTION}/paired_action_bootstrap_quality_v2.json"
  "${SELECTION}/paired_rollout_bootstrap_quality_v2.json"
  "${SELECTION}/paired_writeback_bootstrap_quality_v2.json"
  "${SELECTION}/promotion_decision.json"
)
while true; do
  ready=true
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || ready=false
  done
  if [[ "$ready" == "true" ]] && \
     [[ "$(cat "${SELECTION}/paired_comparison_quality_v2.exit_code")" == "0" ]]; then
    break
  fi
  sleep 15
done

cd "$PROJECT_ROOT"
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" scripts/assess_agent_dpo_paired.py \
  "${SELECTION}/paired_action_bootstrap_quality_v2.json" \
  "${SELECTION}/paired_rollout_bootstrap_quality_v2.json" \
  "${SELECTION}/promotion_decision.json" \
  "$OUTPUT" \
  --writeback-comparison "${SELECTION}/paired_writeback_bootstrap_quality_v2.json"
