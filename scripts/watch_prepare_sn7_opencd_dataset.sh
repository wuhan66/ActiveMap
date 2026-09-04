#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
OUTPUT="${STORAGE_ROOT}/processed/sn7_v1/opencd_image_map_20260726"
MODALITY_RESULT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba/full_weight5_modality_comparison_20260726.json"
OPENCD_RUNTIME="${STORAGE_ROOT}/logs/opencd_setup_20260726/runtime_audit.json"
POLL_SECONDS="${POLL_SECONDS:-120}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

until [[ -s "${MODALITY_RESULT}" && -s "${OPENCD_RUNTIME}" ]]; do
  echo "Waiting for modality training completion and Open-CD runtime audit."
  sleep "${POLL_SECONDS}"
done
if [[ -e "${OUTPUT}" ]]; then
  echo "Refusing to overwrite Open-CD dataset: ${OUTPUT}" >&2
  exit 1
fi

"${PYTHON}" scripts/prepare_sn7_opencd_dataset.py "${MANIFEST}" "${OUTPUT}"
