#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: queue_muno21_direct_vlm_sft_followup.sh GPU SEED UPSTREAM_RUN MODEL RUN_FAMILY}"
SEED="${2:?usage: queue_muno21_direct_vlm_sft_followup.sh GPU SEED UPSTREAM_RUN MODEL RUN_FAMILY}"
UPSTREAM_RUN="${3:?usage: queue_muno21_direct_vlm_sft_followup.sh GPU SEED UPSTREAM_RUN MODEL RUN_FAMILY}"
MODEL="${4:?usage: queue_muno21_direct_vlm_sft_followup.sh GPU SEED UPSTREAM_RUN MODEL RUN_FAMILY}"
RUN_FAMILY="${5:?usage: queue_muno21_direct_vlm_sft_followup.sh GPU SEED UPSTREAM_RUN MODEL RUN_FAMILY}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${SEED}"

mkdir -p "${RUN_ROOT}"
exec 7>"${RUN_ROOT}/.followup.lock"
flock -n 7 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0

while [[ ! -s "${UPSTREAM_RUN}/COMPLETE.json" ]]; do
  [[ ! -s "${UPSTREAM_RUN}/FAILED.json" ]] || exit 3
  sleep 60
done
while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

cd "${PROJECT_ROOT}"
exec env GPU="${GPU}" SEED="${SEED}" MODEL="${MODEL}" \
  RUN_FAMILY="${RUN_FAMILY}" bash scripts/watch_train_muno21_direct_vlm_sft.sh
