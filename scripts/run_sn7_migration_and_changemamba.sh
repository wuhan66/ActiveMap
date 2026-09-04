#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
MIGRATION_LOG="${STORAGE_ROOT}/logs/migrate_sn7_tmux_20260726.log"
TRAIN_QUEUE_LOG="${STORAGE_ROOT}/logs/changemamba_after_migration_tmux_20260726.log"

export KEY_COMMENT="${KEY_COMMENT:-activemap-sn7-transfer-20260726}"

bash "${PROJECT_ROOT}/scripts/migrate_sn7_nts_to_hdpi.sh" \
  > "${MIGRATION_LOG}" 2>&1

bash "${PROJECT_ROOT}/scripts/launch_sn7_changemamba_after_migration.sh" \
  > "${TRAIN_QUEUE_LOG}" 2>&1
