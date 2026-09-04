#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
THRESHOLD_TAG="${MUNO21_V12_THRESHOLD_TAG:-threshold009}"
RUN="${STORAGE_ROOT}/runs/agent/muno21_v12_${THRESHOLD_TAG}_writeback_v1"
OUTPUT="${STORAGE_ROOT}/artifacts/paper_evidence/muno21_v12_${THRESHOLD_TAG}_writeback_4seed_v1"
SEEDS=(20260822 20260823 20260824 20260825)

mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.watch.lock"
flock -n 9 || exit 0
exec >>"${OUTPUT}/watcher.log" 2>&1
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0

echo "[$(date -Is)] waiting for ${THRESHOLD_TAG} writebacks"
for seed in "${SEEDS[@]}"; do
  while [[ ! -s "${RUN}/seed${seed}/COMPLETE.json" ]]; do
    sleep 60
  done
done

cd "${PROJECT_ROOT}"
MUNO21_V12_THRESHOLD_TAG="${THRESHOLD_TAG}" \
  bash scripts/aggregate_muno21_v12_threshold009_writebacks.sh
echo "[$(date -Is)] ${THRESHOLD_TAG} writeback aggregation complete"
