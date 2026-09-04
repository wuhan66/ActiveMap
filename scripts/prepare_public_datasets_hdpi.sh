#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
export ACTIVEMAP_PYTHON="${ACTIVEMAP_GIS_ENV}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${ACTIVEMAP_LOG_ROOT}/dataset_preparation"
STATUS_FILE="${LOG_DIR}/status.tsv"
LOCK_FILE="${LOG_DIR}/preparation.lock"
mkdir -p "${LOG_DIR}" "${ACTIVEMAP_PROCESSED_ROOT}" \
  "${ACTIVEMAP_STORAGE_ROOT}/manifests" "${ACTIVEMAP_STORAGE_ROOT}/splits"

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "Another dataset preparation process holds ${LOCK_FILE}" >&2
  exit 9
fi

record_status() {
  printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$1" "$2" | tee -a "${STATUS_FILE}"
}

audit_passed() {
  [[ -s "$1" ]] && "${ACTIVEMAP_PYTHON}" -c \
    'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("passed") else 1)' \
    "$1"
}

cd "${PROJECT_ROOT}"

sn7_v4="${ACTIVEMAP_PROCESSED_ROOT}/sn7_v1/updater_v4_cap20"
if ! audit_passed "${sn7_v4}/audit.json"; then
  record_status sn7 preparing
  bash scripts/prepare_sn7.sh \
    "${ACTIVEMAP_DATASET_ROOT}/sn7/train/train" \
    "${ACTIVEMAP_PROCESSED_ROOT}/sn7_v1" \
    "${ACTIVEMAP_STORAGE_ROOT}/manifests" \
    "${ACTIVEMAP_STORAGE_ROOT}/splits"
  SN7_MANIFEST="${ACTIVEMAP_STORAGE_ROOT}/manifests/sn7_split.parquet" \
  SN7_V4_OUTPUT="${sn7_v4}" \
    bash scripts/build_sn7_v4_cap20.sh
fi
audit_passed "${sn7_v4}/audit.json"
record_status sn7 pending_manual_qc

record_status muno21 preparing
bash scripts/prepare_muno21_updater.sh
audit_passed "${ACTIVEMAP_PROCESSED_ROOT}/muno21_v2/updater/audit.json"
record_status muno21 pending_manual_qc

record_status inria preparing
bash scripts/prepare_inria_segmentation.sh
audit_passed "${ACTIVEMAP_PROCESSED_ROOT}/inria_v1/segmentation/audit.json"
record_status inria pending_manual_qc

touch "${LOG_DIR}/PENDING_MANUAL_QC"
record_status all pending_manual_qc

mkdir -p "${ACTIVEMAP_STORAGE_ROOT}/artifacts/paper_preflight"
"${ACTIVEMAP_PYTHON}" scripts/audit_paper_experiment_registry.py \
  configs/experiments/paper_registry.yaml "${ACTIVEMAP_STORAGE_ROOT}" \
  "${ACTIVEMAP_STORAGE_ROOT}/artifacts/paper_preflight/current.json" \
  --output-markdown \
  "${ACTIVEMAP_STORAGE_ROOT}/artifacts/paper_preflight/current.md"
