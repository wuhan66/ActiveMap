#!/usr/bin/env bash
set -euo pipefail

STORE="${STORE:-/home/wh/ActiveMap}"
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
FOLLOWUP_LOGS="${STORE}/logs/spacenet8_opencd_smoke80_followup_v1"

for name in \
  pre_post_seed20260803 \
  pre_post_seed20260804 \
  pre_post_seed20260805 \
  pre_post_weight10_seed20260802 \
  pre_post_weight30_seed20260802; do
  pid="$(cat "${FOLLOWUP_LOGS}/${name}.pid")"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 20
  done
  test -s "${STORE}/runs/spacenet8_germany/opencd_smoke80_v1/${name}/summary.json"
done

DATA="${STORE}/processed/spacenet8_germany/aligned_full202_v1" \
ROOT="${STORE}/runs/spacenet8_germany/opencd_full202_v1" \
LOGS="${STORE}/logs/spacenet8_opencd_full202_v1" \
MATRIX=full PROJECT="${PROJECT}" STORE="${STORE}" \
  bash "${PROJECT}/scripts/launch_spacenet8_opencd_followup_five_gpu_hdpi.sh"
