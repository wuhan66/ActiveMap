#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
DPO_RUN="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_v2_persistent_seed20260821}"
BASELINE_RUN="${MUNO21_BASELINE_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_policy_baselines_v1}"
SELECTION="${DPO_RUN}/evaluation/selection"
STATUS="${SELECTION}/policy_baseline_comparison.exit_code"
DPO_WRITEBACK="${DPO_RUN}/evaluation/safety-dpo-best/writeback/writeback.jsonl"

mkdir -p "$SELECTION"
rm -f "$STATUS"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS"' EXIT
while [[ ! -s "${BASELINE_RUN}/watcher.exit_code" ]] || \
      [[ ! -s "$DPO_WRITEBACK" ]]; do
  sleep 30
done
[[ "$(cat "${BASELINE_RUN}/watcher.exit_code")" == "0" ]] || exit 2

cd "$PROJECT_ROOT"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
for baseline in generic_selector edit_conditioned_selector; do
  "$PYTHON" scripts/compare_agent_writebacks.py \
    "${BASELINE_RUN}/writeback/${baseline}/writeback.jsonl" \
    "$DPO_WRITEBACK" \
    "${SELECTION}/paired_writeback_agent_vs_${baseline}.json" \
    --bootstrap 2000 --seed 20260821
done

echo "[$(date --iso-8601=seconds)] Agent versus learned-policy writeback comparisons complete"
