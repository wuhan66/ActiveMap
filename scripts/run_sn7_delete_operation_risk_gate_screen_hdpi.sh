#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
V6_ROOT="${V6_WRITEBACK_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v6_evidence_value_writeback_v3_final}"
OUTPUT_ROOT="${DELETE_RISK_GATE_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_delete_operation_risk_gate_screen_20260830_v1}"
SEEDS=(20260817 20260818 20260819)

[[ -x "${PYTHON}" ]] || { echo "missing ActiveMap Python: ${PYTHON}" >&2; exit 1; }
[[ -d "${PROJECT_ROOT}" && -d "${V6_ROOT}" ]] || { echo "missing screen inputs" >&2; exit 1; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}/logs"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

for seed in "${SEEDS[@]}"; do
  train_root="${V6_ROOT}/raw_writebacks/seed${seed}/train"
  validation_root="${V6_ROOT}/raw_writebacks/seed${seed}/val"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/screen_sn7_delete_operation_risk_gate.py" \
    "${OUTPUT_ROOT}/seed${seed}.json" \
    --seed "${seed}" \
    --train-record "direct=${train_root}/direct/writeback.jsonl" \
    --train-record "selected=${train_root}/selected/writeback.jsonl" \
    --validation-record "direct=${validation_root}/direct/writeback.jsonl" \
    --validation-record "selected=${validation_root}/selected/writeback.jsonl" \
    >"${OUTPUT_ROOT}/logs/seed${seed}.log" 2>&1
done

printf '%s\n' \
  '{' \
  '  "status": "complete",' \
  '  "protocol": "train-only multi-feature DELETE risk-gate screen",' \
  '  "validation_used_for_threshold_tuning": false,' \
  '  "test_assets_read": false' \
  '}' >"${OUTPUT_ROOT}/status.json"

echo "SN7 DELETE risk-gate screen complete: ${OUTPUT_ROOT}"
