#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
GPU="${SAM_ROAD_GPU:-3}"
ENV_PREFIX="${SAM_ROAD_ENV:-/home/wh/yes/envs/activemap-sam-road}"
RUN_NAME="${SAM_ROAD_RUN_NAME:-muno21_sam_road_head_seed20260716_v1}"
RUN_DIR="${SAM_ROAD_RUN_DIR:-${STORAGE_ROOT}/runs/semantic/${RUN_NAME}}"
LOG_PATH="${SAM_ROAD_LOG:-${STORAGE_ROOT}/logs/${RUN_NAME}.log}"

if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run: $RUN_DIR" >&2
  exit 2
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
  echo "Missing SAM-Road environment: ${ENV_PREFIX}" >&2
  exit 2
fi

IFS=',' read -r utilization memory_used < <(
  nvidia-smi --id="$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( utilization > 10 || memory_used > 1024 )); then
  echo "GPU ${GPU} is not idle: utilization=${utilization}%, memory=${memory_used} MiB" >&2
  exit 3
fi

mkdir -p "$(dirname "$LOG_PATH")"
cd "$PROJECT_ROOT"
export ACTIVEMAP_DISABLE_FROZEN_TEST=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="$GPU"

exec "${ENV_PREFIX}/bin/python" scripts/train_muno21_sam_road_head.py \
  "${STORAGE_ROOT}/processed/muno21_v2/rsprompter_road_v2_rle" \
  "${STORAGE_ROOT}/external/sam_road" \
  "${STORAGE_ROOT}/external/sam_road/config/toponet_vitb_256_spacenet.yaml" \
  "${STORAGE_ROOT}/models/sam_road/spacenet_vitb_256_e10.ckpt" \
  "${STORAGE_ROOT}/models/sam_road/sam_vit_b_01ec64.pth" \
  "$RUN_DIR" \
  --calibration-aoi atlanta \
  --device cuda:0 \
  --seed 20260716 \
  --batch-size 8 \
  --workers 4 \
  --max-epochs 30 \
  --patience 6 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --max-positive-weight 12.0 \
  2>&1 | tee "$LOG_PATH"
