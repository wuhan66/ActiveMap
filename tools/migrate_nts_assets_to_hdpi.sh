#!/usr/bin/env bash
set -euo pipefail

SOURCE_HOST="${SOURCE_HOST:-nts-server.harmo.icu}"
SOURCE_PORT="${SOURCE_PORT:-50053}"
SOURCE_USER="${SOURCE_USER:-wh}"
SOURCE_KEY="${SOURCE_KEY:-/home/wh/.ssh/activemap_nts_transfer}"
STAGING_ROOT="${STAGING_ROOT:-/home/wh/projects/activemap-v1-nts-sync-20260716}"
TARGET_STORAGE="${TARGET_STORAGE:-/home/wh/ActiveMap}"
LOG_ROOT="${TARGET_STORAGE}/migration"
SSH_COMMAND="ssh -i ${SOURCE_KEY} -p ${SOURCE_PORT} -o BatchMode=yes"

mkdir -p \
  "$STAGING_ROOT" \
  "${TARGET_STORAGE}/processed/muno21_v2/rsprompter_road_v2_rle" \
  "${TARGET_STORAGE}/external/sam_road" \
  "${TARGET_STORAGE}/models/sam_road" \
  "$LOG_ROOT"

rsync -a --partial --info=progress2 \
  --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' \
  -e "$SSH_COMMAND" \
  "${SOURCE_USER}@${SOURCE_HOST}:/home/wh/projects/activemap-v1-joint-debug/" \
  "${STAGING_ROOT}/"

rsync -a --partial --info=progress2 -e "$SSH_COMMAND" \
  "${SOURCE_USER}@${SOURCE_HOST}:/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/rsprompter_road_v2_rle/" \
  "${TARGET_STORAGE}/processed/muno21_v2/rsprompter_road_v2_rle/"

rsync -a --partial --info=progress2 -e "$SSH_COMMAND" \
  "${SOURCE_USER}@${SOURCE_HOST}:/mnt/mydisk/wh/ActiveMap/external/sam_road/" \
  "${TARGET_STORAGE}/external/sam_road/"

rsync -a --partial --info=progress2 -e "$SSH_COMMAND" \
  "${SOURCE_USER}@${SOURCE_HOST}:/mnt/mydisk/wh/ActiveMap/models/sam_road/" \
  "${TARGET_STORAGE}/models/sam_road/"

sha256sum \
  "${TARGET_STORAGE}/processed/muno21_v2/rsprompter_road_v2_rle/annotations/train.json" \
  "${TARGET_STORAGE}/processed/muno21_v2/rsprompter_road_v2_rle/annotations/val.json" \
  "${TARGET_STORAGE}/models/sam_road/spacenet_vitb_256_e10.ckpt" \
  "${TARGET_STORAGE}/models/sam_road/sam_vit_b_01ec64.pth" \
  > "${LOG_ROOT}/nts_to_hdpi_20260716.sha256"

{
  printf 'completed_at=%s\n' "$(date --iso-8601=seconds)"
  printf 'source=%s@%s:%s\n' "$SOURCE_USER" "$SOURCE_HOST" "$SOURCE_PORT"
  printf 'staging_root=%s\n' "$STAGING_ROOT"
  printf 'target_storage=%s\n' "$TARGET_STORAGE"
  du -sh \
    "$STAGING_ROOT" \
    "${TARGET_STORAGE}/processed/muno21_v2/rsprompter_road_v2_rle" \
    "${TARGET_STORAGE}/external/sam_road" \
    "${TARGET_STORAGE}/models/sam_road"
} > "${LOG_ROOT}/nts_to_hdpi_20260716.complete"
