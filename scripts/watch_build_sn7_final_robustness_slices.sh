#!/usr/bin/env bash
set -euo pipefail

ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ROOT}/envs/activemap-agent/bin/python"
RUN="${ROOT}/runs/sn7_active_catalog"
CAUSAL="${RUN}/final_tool_belief_causal_v1"
EPISODES="${ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full/closed_loop_v1/episodes_val.jsonl"
OUTPUT="${RUN}/final_robustness_slices_v1"
POLL_SECONDS="${POLL_SECONDS:-60}"

until [[ -s "${CAUSAL}/manifest.json" ]]; do sleep "${POLL_SECONDS}"; done
cd "${REPO}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
"${PYTHON}" scripts/aggregate_active_catalog_robustness_slices.py \
  "${EPISODES}" "${OUTPUT}" \
  --pair "20260717=${CAUSAL}/seed20260717/no_tool/evaluation/traces.jsonl,${CAUSAL}/seed20260717/selective/evaluation/traces.jsonl" \
  --pair "20260718=${CAUSAL}/seed20260718/no_tool/evaluation/traces.jsonl,${CAUSAL}/seed20260718/selective/evaluation/traces.jsonl" \
  --pair "20260719=${CAUSAL}/seed20260719/no_tool/evaluation/traces.jsonl,${CAUSAL}/seed20260719/selective/evaluation/traces.jsonl" \
  --expected-records 512 --repetitions 5000 --seed 20260729
