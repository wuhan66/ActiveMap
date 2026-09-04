#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/agent/muno21_qwen3_8b_balanced_tool_sft_seed20260908"
OUTPUT="${STORAGE_ROOT}/artifacts/paper_evidence/muno21_qwen3_8b_capacity_trajectory_v1"

mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${OUTPUT}/capacity_trajectory.md" ]] || exit 0
exec >>"${OUTPUT}/watcher.log" 2>&1

for step in 100 200 300 400; do
  while [[ ! -s "${RUN}/evaluation/capacity_trajectory_v1/checkpoint-${step}/COMPLETE.json" ]]; do
    sleep 30
  done
done
while [[ ! -s "${RUN}/evaluation/capacity_static_v1/COMPLETE.json" ]]; do
  sleep 30
done

cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/summarize_muno21_qwen3_8b_capacity_trajectory.py \
  "${RUN}" "${OUTPUT}"
echo "[$(date -Is)] Qwen3-8B capacity trajectory complete"
