#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
V6_ROOT="${V6_WRITEBACK_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v6_evidence_value_writeback_v3_final}"
SAFE_ROOT="${OPERATION_SAFE_COMMIT_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_operation_safe_commit_screen_20260830_v2}"
OUTPUT_ROOT="${OPERATION_FOLLOWUP_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_operation_followup_audits_20260830_v5}"
SEEDS=(20260817 20260818 20260819)

[[ -x "${PYTHON}" ]] || { echo "missing ActiveMap Python: ${PYTHON}" >&2; exit 1; }
[[ -d "${PROJECT_ROOT}" && -d "${V6_ROOT}" && -d "${SAFE_ROOT}" ]] || {
  echo "missing operation-audit inputs" >&2
  exit 1
}
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "refusing to overwrite ${OUTPUT_ROOT}" >&2; exit 1; }

mkdir -p "${OUTPUT_ROOT}/logs"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

bundle_args=()
delete_args=()
for seed in "${SEEDS[@]}"; do
  if [[ "${seed}" == "20260817" ]]; then
    states="${DATA_ROOT}/selector_states_headroom_val_v5b.jsonl"
  else
    states="${DATA_ROOT}/selector_states_headroom_val_v5b_seed${seed}.jsonl"
  fi
  rollout="${V6_ROOT}/rollouts/seed${seed}_val/selected_rollouts.jsonl"
  raw_val="${V6_ROOT}/raw_writebacks/seed${seed}/val/selected/writeback.jsonl"
  safe_val="${SAFE_ROOT}/safe_writebacks/seed${seed}/selected/writeback.jsonl"
  bundle_args+=(--bundle "${seed}:${states}:${rollout}:${raw_val}:${safe_val}")

  raw_train="${V6_ROOT}/raw_writebacks/seed${seed}/train"
  delete_args+=(--record "seed${seed}_direct=${raw_train}/direct/writeback.jsonl")
  delete_args+=(--record "seed${seed}_selected=${raw_train}/selected/writeback.jsonl")
done

"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_sn7_counterfactual_runtime_alignment.py" \
  "${OUTPUT_ROOT}/counterfactual_runtime_alignment" "${bundle_args[@]}" \
  >"${OUTPUT_ROOT}/logs/counterfactual_runtime_alignment.log" 2>&1

"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_sn7_delete_confidence_separability.py" \
  "${OUTPUT_ROOT}/delete_confidence_separability" "${delete_args[@]}" \
  --maximum-harmful-accept-rate 0.01 \
  >"${OUTPUT_ROOT}/logs/delete_confidence_separability.log" 2>&1

printf '%s\n' \
  '{' \
  '  "status": "complete",' \
  '  "protocol": "SN7 operation follow-up diagnostics",' \
  '  "validation_used_for_threshold_tuning": false,' \
  '  "test_assets_read": false' \
  '}' >"${OUTPUT_ROOT}/status.json"

echo "SN7 operation follow-up audits complete: ${OUTPUT_ROOT}"
