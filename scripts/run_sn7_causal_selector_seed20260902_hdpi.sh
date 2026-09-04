#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${GPU:-3}"
SOURCE="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_selector_runtime_grid_budget3_v1/selector_states_train_val_step0_online_runtime_grid_budget3_v1.jsonl"
OUTPUT="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_selector_runtime_grid_budget3_causal_v1/selector_states_train_val_causal_v1.jsonl"
TRAIN_EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl"
VAL_EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
RUN="${STORAGE_ROOT}/runs/selector/sn7_causal_online_runtime_grid_v1_seed20260902"
CONFIG="${PROJECT_ROOT}/configs/selector/sn7_causal_online_runtime_grid_v1.yaml"

[[ -x "${PYTHON}" && -f "${SOURCE}" && -f "${TRAIN_EPISODES}" && -f "${VAL_EPISODES}" ]] || {
  echo "missing Python or causal selector inputs" >&2
  exit 2
}
[[ ! -e "${OUTPUT}" && ! -e "${RUN}" ]] || {
  echo "causal selector output already exists" >&2
  exit 2
}
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" scripts/build_causal_selector_manifest.py "${SOURCE}" "${OUTPUT}" \
  --episodes "${TRAIN_EPISODES}" --episodes "${VAL_EPISODES}"
"${PYTHON}" scripts/audit_selector_states.py "${OUTPUT}" \
  --output "${OUTPUT%.jsonl}.audit.json"
"${PYTHON}" scripts/audit_selector_utility_structure.py "${OUTPUT}" \
  --output "${OUTPUT%.jsonl}.utility_audit.json"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m activemap.cli train-selector \
  "${CONFIG}" --output "${RUN}"

echo "completed causal selector seed 20260902 under ${RUN}"
