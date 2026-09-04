#!/usr/bin/env bash
set -euo pipefail

SOURCE_HOST="${SOURCE_HOST:-nts-server.harmo.icu}"
SOURCE_PORT="${SOURCE_PORT:-50053}"
SOURCE_USER="${SOURCE_USER:-wh}"
SOURCE_KEY="${SOURCE_KEY:-/home/wh/.ssh/activemap_nts_transfer}"
SOURCE_STORAGE="${SOURCE_STORAGE:-/mnt/mydisk/wh/ActiveMap}"
TARGET_STORAGE="${TARGET_STORAGE:-/home/wh/ActiveMap}"
TARGET_REPO="${TARGET_REPO:-/home/wh/projects/activemap-v1}"
RSYNC_RSH="ssh -i ${SOURCE_KEY} -p ${SOURCE_PORT} -o BatchMode=yes"

mkdir -p \
  "${TARGET_STORAGE}/processed/muno21_v2/updater" \
  "${TARGET_STORAGE}/processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools" \
  "${TARGET_STORAGE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools" \
  "${TARGET_STORAGE}/runs/semantic/muno21_sam_road_head_seed20260716_v1/checkpoints" \
  "${TARGET_STORAGE}/data-versions" \
  "${TARGET_STORAGE}/migration"

rsync -a --partial --info=progress2 -e "$RSYNC_RSH" \
  "${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_STORAGE}/processed/muno21_v2/updater/" \
  "${TARGET_STORAGE}/processed/muno21_v2/updater/"

rsync -a --partial --info=progress2 -e "$RSYNC_RSH" \
  "${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_STORAGE}/processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools/" \
  "${TARGET_STORAGE}/processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools/"

rsync -a --partial --info=progress2 -e "$RSYNC_RSH" \
  "${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_STORAGE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/" \
  "${TARGET_STORAGE}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools/"

rsync -a --partial --info=progress2 -e "$RSYNC_RSH" \
  "${SOURCE_USER}@${SOURCE_HOST}:${SOURCE_STORAGE}/runs/semantic/muno21_sam_road_head_seed20260716_v1/checkpoints/best_full.ckpt" \
  "${TARGET_STORAGE}/runs/semantic/muno21_sam_road_head_seed20260716_v1/checkpoints/"

bash "${TARGET_REPO}/tools/inventory_canonical_data.sh" \
  "$TARGET_STORAGE" "${TARGET_STORAGE}/data-versions/current.tsv"

date --iso-8601=seconds > "${TARGET_STORAGE}/migration/muno21_training_assets_sync.complete"
