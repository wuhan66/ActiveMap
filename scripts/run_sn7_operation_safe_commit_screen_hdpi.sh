#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
V6_ROOT="${V6_WRITEBACK_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v6_evidence_value_writeback_v3_final}"
OUTPUT_ROOT="${OPERATION_SAFE_COMMIT_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_operation_safe_commit_screen_20260830_v2}"
MAX_FALSE_EDIT_RATE="${MAX_FALSE_EDIT_RATE:-0.01}"
SEEDS=(20260817 20260818 20260819)

[[ -x "${PYTHON}" ]] || { echo "missing ActiveMap Python: ${PYTHON}" >&2; exit 1; }
[[ -d "${PROJECT_ROOT}" && -d "${V6_ROOT}" ]] || { echo "missing project inputs" >&2; exit 1; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}/calibration" "${OUTPUT_ROOT}/logs"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

for seed in "${SEEDS[@]}"; do
  train_root="${V6_ROOT}/raw_writebacks/seed${seed}/train"
  val_root="${V6_ROOT}/raw_writebacks/seed${seed}/val"
  calibration="${OUTPUT_ROOT}/calibration/seed${seed}.json"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/calibrate_sn7_operation_safe_commit.py" \
    "${calibration}" \
    --record "direct=${train_root}/direct/writeback.jsonl" \
    --record "selected=${train_root}/selected/writeback.jsonl" \
    --maximum-false-edit-rate "${MAX_FALSE_EDIT_RATE}" \
    >"${OUTPUT_ROOT}/logs/calibrate_seed${seed}.log" 2>&1

  for policy in direct selected; do
    "${PYTHON}" "${PROJECT_ROOT}/scripts/apply_sn7_operation_safe_commit.py" \
      "${val_root}/${policy}/writeback.jsonl" "${calibration}" \
      "${OUTPUT_ROOT}/safe_writebacks/seed${seed}/${policy}" \
      --policy "${policy}" \
      >"${OUTPUT_ROOT}/logs/apply_${policy}_seed${seed}.log" 2>&1
  done
done

aggregate_args=()
for seed in "${SEEDS[@]}"; do
  raw="${V6_ROOT}/raw_writebacks/seed${seed}/val"
  safe="${OUTPUT_ROOT}/safe_writebacks/seed${seed}"
  aggregate_args+=(--record "${seed}:direct_commit=${raw}/direct/writeback.jsonl")
  aggregate_args+=(--record "${seed}:direct_safe_commit=${safe}/direct/writeback.jsonl")
  aggregate_args+=(--record "${seed}:selected_commit=${raw}/selected/writeback.jsonl")
  aggregate_args+=(--record "${seed}:selected_safe_commit=${safe}/selected/writeback.jsonl")
done
"${PYTHON}" "${PROJECT_ROOT}/scripts/aggregate_sn7_v5_factorial_writebacks.py" \
  "${OUTPUT_ROOT}/three_seed_factorial_summary.json" "${aggregate_args[@]}" \
  >"${OUTPUT_ROOT}/logs/aggregate.log" 2>&1

bundle_args=()
for seed in "${SEEDS[@]}"; do
  if [[ "${seed}" == "20260817" ]]; then
    states="${DATA_ROOT}/selector_states_headroom_val_v5b.jsonl"
  else
    states="${DATA_ROOT}/selector_states_headroom_val_v5b_seed${seed}.jsonl"
  fi
  bundle_args+=(--bundle "${seed}:${states}:${V6_ROOT}/rollouts/seed${seed}_val/selected_rollouts.jsonl:${V6_ROOT}/raw_writebacks/seed${seed}/val/selected/writeback.jsonl:${OUTPUT_ROOT}/safe_writebacks/seed${seed}/selected/writeback.jsonl")
done
"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_sn7_operation_support_layers.py" \
  "${OUTPUT_ROOT}/operation_support_audit" "${bundle_args[@]}" \
  >"${OUTPUT_ROOT}/logs/operation_support_audit.log" 2>&1

"${PYTHON}" - "${OUTPUT_ROOT}/status.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "complete",
            "protocol": "train-only operation-conditioned Safe Commit screen",
            "validation_used_for_threshold_tuning": False,
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

echo "operation-conditioned Safe Commit screen complete: ${OUTPUT_ROOT}"
