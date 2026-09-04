#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
MODALITY_AUDIT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba/full_weight5_modality_aoi_bootstrap_20260726.json"
OUTPUT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba/candidate_localization_audit_20260726.json"
LOG="${STORAGE_ROOT}/logs/sn7_candidate_localization_audit_20260726.log"
POLL_SECONDS="${POLL_SECONDS:-120}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
while [[ ! -s "${MODALITY_AUDIT}" ]]; do
  echo "Waiting for modality audit to release SN7 storage bandwidth."
  sleep "${POLL_SECONDS}"
done
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite candidate-localization audit: ${OUTPUT}" >&2
  exit 1
fi
"${PYTHON}" scripts/audit_sn7_candidate_localization.py \
  "${MANIFEST}" "${OUTPUT}" > "${LOG}" 2>&1
