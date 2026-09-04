#!/usr/bin/env bash
set -euo pipefail

echo "Deprecated: this V5 selector has no acquired non-KEEP support; use run_sn7_v6_counterfactual_alignment_smoke_hdpi.sh" >&2
exit 2

# Validation-only parity smoke. It reuses the registered V5 updater and selector
# and changes only the execution semantics from all-evidence fusion to the
# standalone candidate outcome used by counterfactual supervision.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
SOURCE_RUN_ROOT="${V5_MATCHED_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_matched_nonkeep_2x2_v2}"
RUN_ROOT="${V5_ALIGNMENT_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_counterfactual_alignment_smoke_seed20260817}"
SEED="${UPDATER_SEED:-20260817}"
GPU_ID="${GPU_ID:-0}"

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite smoke root: ${RUN_ROOT}" >&2; exit 1; }
active="$(nvidia-smi -i "${GPU_ID}" --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${GPU_ID} is occupied: ${active}" >&2; exit 1; }

EPISODES="${DATA_ROOT}/episodes_trainval_v5.jsonl"
STATES="${DATA_ROOT}/selector_states_headroom_val_v5b.jsonl"
CHECKPOINT="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${SEED}/best_quality.pt"
SELECTOR="${SOURCE_RUN_ROOT}/selectors/seed${SEED}/best.pt"
AUTHORIZATION="${SOURCE_RUN_ROOT}/authorization/three_seed_headroom_authorization.json"
for path in "${EPISODES}" "${STATES}" "${CHECKPOINT}" "${SELECTOR}" "${AUTHORIZATION}"; do
  [[ -f "${path}" ]] || { echo "Missing smoke input: ${path}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.queue.lock"
flock -n 9 || { echo "Counterfactual-alignment smoke is already active" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ROLLOUT_DIR="${RUN_ROOT}/rollouts"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" \
  "${PROJECT_ROOT}/scripts/build_sn7_v5_matched_rollouts.py" \
  "${STATES}" "${ROLLOUT_DIR}" \
  --authorization "${AUTHORIZATION}" \
  --updater-checkpoint "${CHECKPOINT}" \
  --updater-seed "${SEED}" \
  --selector-checkpoint "${SELECTOR}" \
  --split val --device cuda:0 \
  --writeback-evidence-mode counterfactual_aligned \
  >"${RUN_ROOT}/build_rollouts.log" 2>&1

WRITEBACK_DIR="${RUN_ROOT}/selected_writeback"
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON}" \
  "${PROJECT_ROOT}/scripts/evaluate_agent_map_writeback.py" \
  "${CHECKPOINT}" "${EPISODES}" "${ROLLOUT_DIR}/selected_rollouts.jsonl" \
  "${WRITEBACK_DIR}" \
  --device cuda --split val --image-size 128 --threshold 0.5 \
  --protocol-name sn7-v5-counterfactual-aligned-smoke-v1 \
  >"${RUN_ROOT}/writeback.log" 2>&1

"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_sn7_counterfactual_runtime_alignment.py" \
  "${RUN_ROOT}/alignment_audit" \
  --bundle "${SEED}:${STATES}:${ROLLOUT_DIR}/selected_rollouts.jsonl:${WRITEBACK_DIR}/writeback.jsonl:${WRITEBACK_DIR}/writeback.jsonl" \
  >"${RUN_ROOT}/alignment_audit.log" 2>&1

"${PYTHON}" - "${RUN_ROOT}/alignment_audit/summary.json" "${RUN_ROOT}/gate.json" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
reshape = next(
    row
    for row in summary["groups"]
    if row["seed"] == "all" and row["operation"] == "RESHAPE"
)
passed = (
    reshape["mean_actual_raster_gain"] >= 0.0
    and reshape["sign_agreement_rate"] >= 0.50
    and reshape["exact_oracle_runtime_positive_rate"] > 0.0
)
payload = {
    "schema_version": "sn7-v5-counterfactual-alignment-smoke-gate-v1",
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
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY

echo "Counterfactual-alignment smoke complete: ${RUN_ROOT}"
