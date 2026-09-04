#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
CONCAT="${STORAGE_ROOT}/runs/updater/concat_unet_three_seed_val_20260729/paper_artifacts/aggregate.json"
EXTERNAL="${PROJECT_ROOT}/docs/SN7_BAN_VS_CHANGEMAMBA_BOOTSTRAP_20260727.json"
OUTPUT="${STORAGE_ROOT}/paper/tables/sn7_perception_baselines_v1"
LOG="${STORAGE_ROOT}/logs/sn7_perception_baseline_table_20260729.log"

mkdir -p "$(dirname "${LOG}")"
exec >>"${LOG}" 2>&1
cd "${PROJECT_ROOT}"
echo "[$(date --iso-8601=seconds)] waiting for concat U-Net aggregate"
until [[ -s "${CONCAT}" ]]; do
  sleep 120
done

"${PYTHON}" scripts/export_sn7_perception_baseline_table.py \
  "${OUTPUT}" \
  --concat "${CONCAT}" \
  --external-comparison "${EXTERNAL}"
echo "[$(date --iso-8601=seconds)] SN7 perception table complete"
