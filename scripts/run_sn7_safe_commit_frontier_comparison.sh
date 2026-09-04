#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-opencd/bin/python"
CHANGE_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
BAN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
TRANSFER_ROOT="${STORAGE_ROOT}/runs/safe_commit_cross_backend_20260727"
OUTPUT_ROOT="${STORAGE_ROOT}/runs/safe_commit_frontiers_20260727"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

frontier() {
  local name="$1"
  shift
  "${PYTHON}" scripts/analyze_sn7_changemamba_commit_frontier.py \
    "${OUTPUT_ROOT}/${name}.json" "$@" \
    --threshold-count 101 \
    --output-csv "${OUTPUT_ROOT}/${name}.csv" \
    --output-markdown "${OUTPUT_ROOT}/${name}.md" \
    >"${OUTPUT_ROOT}/${name}.log" 2>&1
}

frontier changemamba_self \
  "${CHANGE_ROOT}/full_weight5_seed20260725_v1/safe_commit" \
  "${CHANGE_ROOT}/full_weight5_seed20260726_v1/safe_commit" \
  "${CHANGE_ROOT}/full_weight5_seed20260727_v1/safe_commit"
frontier ban_self \
  "${BAN_ROOT}/safe_commit_seed20260725_v1" \
  "${BAN_ROOT}/safe_commit_seed20260726_v1" \
  "${BAN_ROOT}/safe_commit_seed20260727_v1"
frontier changemamba_to_ban \
  "${TRANSFER_ROOT}/changemamba_to_ban_seed20260725" \
  "${TRANSFER_ROOT}/changemamba_to_ban_seed20260726" \
  "${TRANSFER_ROOT}/changemamba_to_ban_seed20260727"
frontier ban_to_changemamba \
  "${TRANSFER_ROOT}/ban_to_changemamba_seed20260725" \
  "${TRANSFER_ROOT}/ban_to_changemamba_seed20260726" \
  "${TRANSFER_ROOT}/ban_to_changemamba_seed20260727"

"${PYTHON}" scripts/compare_sn7_safe_commit_frontiers.py \
  "${OUTPUT_ROOT}/comparison.json" \
  --frontier changemamba_self "${OUTPUT_ROOT}/changemamba_self.json" \
  --frontier ban_self "${OUTPUT_ROOT}/ban_self.json" \
  --frontier changemamba_to_ban "${OUTPUT_ROOT}/changemamba_to_ban.json" \
  --frontier ban_to_changemamba "${OUTPUT_ROOT}/ban_to_changemamba.json" \
  --output-markdown "${OUTPUT_ROOT}/comparison.md" \
  --output-figure "${OUTPUT_ROOT}/quality_safety_frontiers.png" \
  >"${OUTPUT_ROOT}/comparison.log" 2>&1
