#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_ROOT="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_v2_persistent_seed20260821}"
DPO_LABEL="safety-dpo-best"
SFT_LABEL="sft-checkpoint-200-quality-v2"
SELECTION="${RUN_ROOT}/evaluation/selection"
STATUS_PATH="${SELECTION}/paired_comparison_quality_v2.exit_code"
BASE_ACTIONS="${RUN_ROOT}/evaluation/${SFT_LABEL}/actions/predictions.jsonl"
BASE_ROLLOUTS="${RUN_ROOT}/evaluation/${SFT_LABEL}/rollouts/qwen3_4b_sft.jsonl"
DPO_ACTIONS="${RUN_ROOT}/evaluation/${DPO_LABEL}/actions/predictions.jsonl"
DPO_ROLLOUTS="${RUN_ROOT}/evaluation/${DPO_LABEL}/rollouts/qwen3_4b_sft.jsonl"
BASE_WRITEBACK="${RUN_ROOT}/evaluation/${SFT_LABEL}/writeback/writeback.jsonl"
DPO_WRITEBACK="${RUN_ROOT}/evaluation/${DPO_LABEL}/writeback/writeback.jsonl"

mkdir -p "$SELECTION"
rm -f "$STATUS_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS_PATH"' EXIT

required=(
  "${RUN_ROOT}/evaluation/sft_quality_refresh.exit_code"
  "$BASE_ACTIONS" "$BASE_ROLLOUTS" "$DPO_ACTIONS" "$DPO_ROLLOUTS"
  "$BASE_WRITEBACK" "$DPO_WRITEBACK"
  "${SELECTION}/promotion_decision.json"
)
while true; do
  ready=true
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || ready=false
  done
  if [[ "$ready" == "true" ]] && \
     [[ "$(cat "${RUN_ROOT}/evaluation/sft_quality_refresh.exit_code")" == "0" ]]; then
    break
  fi
  echo "[$(date --iso-8601=seconds)] waiting for matched quality-v2 evaluations"
  sleep 15
done

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"$PYTHON" scripts/validate_quality_rollout_pair.py \
  "$BASE_ROLLOUTS" "$DPO_ROLLOUTS" "${SELECTION}/quality_v2_pair_audit.json"
"$PYTHON" scripts/compare_agent_action_predictions.py \
  "$BASE_ACTIONS" "$DPO_ACTIONS" \
  "${SELECTION}/paired_action_bootstrap_quality_v2.json" \
  --bootstrap 2000 --seed 20260821
"$PYTHON" scripts/compare_agent_rollouts.py \
  "$BASE_ROLLOUTS" "$DPO_ROLLOUTS" \
  "${SELECTION}/paired_rollout_bootstrap_quality_v2.json" \
  --bootstrap 2000 --seed 20260821
"$PYTHON" scripts/compare_agent_writebacks.py \
  "$BASE_WRITEBACK" "$DPO_WRITEBACK" \
  "${SELECTION}/paired_writeback_bootstrap_quality_v2.json" \
  --bootstrap 2000 --seed 20260821

echo "[$(date --iso-8601=seconds)] matched quality-v2 paired comparisons completed"
