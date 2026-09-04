#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_candidate_grpo_formal_3seed_v1_20260807}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_candidate_grpo_formal_3seed_v1_20260807}"

while [[ ! -s "${ROOT}/FORMAL_RESULTS_READY" ]]; do
  sleep 30
done
if [[ ! -s "${ROOT}/formal_result_summary.json" ]]; then
  cd "${PROJECT}"
  export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
  "${PY}" scripts/summarize_muno21_candidate_grpo_formal.py \
    "${ROOT}/three_seed_aggregate.json" \
    "${ROOT}/formal_result_summary.json" \
    "${ROOT}/formal_result.md" \
    >"${LOGROOT}/formal_result_summary.log" 2>&1
fi
date -Is >"${ROOT}/FORMAL_SUMMARY_READY"
