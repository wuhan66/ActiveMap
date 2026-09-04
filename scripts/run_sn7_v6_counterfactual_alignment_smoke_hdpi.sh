#!/usr/bin/env bash
set -euo pipefail

# Validation-only RESHAPE parity smoke over the frozen V6 selected policy.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
V6_ROOT="${V6_WRITEBACK_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v6_evidence_value_writeback_v3_final}"
RUN_ROOT="${V6_ALIGNMENT_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v6_counterfactual_alignment_smoke_seed20260817}"
SEED="${UPDATER_SEED:-20260817}"
GPU_ID="${GPU_ID:-0}"

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite smoke root: ${RUN_ROOT}" >&2; exit 1; }
active="$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${GPU_ID} is occupied: ${active}" >&2; exit 1; }

STATES="${DATA_ROOT}/selector_states_headroom_val_v5b.jsonl"
EPISODES="${DATA_ROOT}/episodes_trainval_v5.jsonl"
CHECKPOINT="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${SEED}/best_quality.pt"
SOURCE_ROLLOUTS="${V6_ROOT}/rollouts/seed${SEED}_val/selected_rollouts.jsonl"
for path in "${STATES}" "${EPISODES}" "${CHECKPOINT}" "${SOURCE_ROLLOUTS}"; do
  [[ -f "${path}" ]] || { echo "Missing smoke input: ${path}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.queue.lock"
flock -n 9 || { echo "V6 alignment smoke is already active" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

SLICE_DIR="${RUN_ROOT}/reshape_slice"
"${PYTHON}" "${PROJECT_ROOT}/scripts/convert_sn7_rollouts_counterfactual_aligned.py" \
  "${STATES}" "${SOURCE_ROLLOUTS}" "${SLICE_DIR}" \
  --operation RESHAPE --require-acquisition \
  >"${RUN_ROOT}/convert.log" 2>&1

WRITEBACK_DIR="${RUN_ROOT}/reshape_writeback"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" \
  "${PROJECT_ROOT}/scripts/evaluate_agent_map_writeback.py" \
  "${CHECKPOINT}" "${EPISODES}" "${SLICE_DIR}/rollouts.jsonl" "${WRITEBACK_DIR}" \
  --device cuda --split val --image-size 128 --threshold 0.5 \
  --protocol-name sn7-v6-counterfactual-aligned-reshape-smoke-v1 \
  >"${RUN_ROOT}/writeback.log" 2>&1

"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_sn7_counterfactual_runtime_alignment.py" \
  "${RUN_ROOT}/alignment_audit" \
  --bundle "${SEED}:${SLICE_DIR}/states.jsonl:${SLICE_DIR}/rollouts.jsonl:${WRITEBACK_DIR}/writeback.jsonl:${WRITEBACK_DIR}/writeback.jsonl" \
  >"${RUN_ROOT}/alignment_audit.log" 2>&1

"${PYTHON}" - "${RUN_ROOT}/alignment_audit/summary.json" "${RUN_ROOT}/gate.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
reshape = next(
    row for row in summary["groups"]
    if row["seed"] == "all" and row["operation"] == "RESHAPE"
)
passed = (
    reshape["mean_actual_raster_gain"] >= 0.0
    and reshape["sign_agreement_rate"] >= 0.50
    and reshape["exact_oracle_runtime_positive_rate"] > 0.0
)
payload = {
    "schema_version": "sn7-v6-counterfactual-alignment-smoke-gate-v1",
    "decision": "expand_to_three_seeds" if passed else "stop_branch",
    "criteria": {
        "mean_actual_raster_gain_nonnegative": reshape["mean_actual_raster_gain"] >= 0.0,
        "sign_agreement_at_least_0p50": reshape["sign_agreement_rate"] >= 0.50,
        "exact_oracle_runtime_positive": reshape["exact_oracle_runtime_positive_rate"] > 0.0,
    },
    "reshape": reshape,
    "split": "val",
    "test_assets_read": False,
}
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "V6 counterfactual-alignment smoke complete: ${RUN_ROOT}"
