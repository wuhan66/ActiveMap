#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_ENV:-${STORAGE_ROOT}/envs/activemap-agent}/bin/python"
SEED="${MUNO21_TOOL_NEED_SEED:-20260831}"
DATA_ROOT="${MUNO21_AGENT_NATURAL_DATA_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools}"
FEATURE_ROOT="${MUNO21_TOOL_NEED_FEATURE_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/structured_tool_need_v11}"
RUN_DIR="${MUNO21_TOOL_NEED_RUN_DIR:-${STORAGE_ROOT}/runs/agent/muno21_tool_need_gate_v11_seed${SEED}}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

for split in train val; do
  output="${FEATURE_ROOT}/${split}"
  if [[ ! -f "${output}/summary.json" ]]; then
    "${PYTHON}" scripts/build_structured_tool_need_features.py \
      "${DATA_ROOT}/${split}/sft_composed.jsonl" "${output}" --split "${split}"
  fi
done

if [[ -e "${RUN_DIR}/summary.json" ]]; then
  echo "MUNO21 v11 Tool-Need gate already complete: ${RUN_DIR}"
  exit 0
fi
if [[ -e "${RUN_DIR}" ]]; then
  echo "refusing partial Tool-Need gate directory: ${RUN_DIR}" >&2
  exit 1
fi

"${PYTHON}" scripts/train_visual_tool_gate.py \
  "${FEATURE_ROOT}/train" "${FEATURE_ROOT}/val" "${RUN_DIR}" \
  --seed "${SEED}" \
  --selection-objective f0_5 \
  --fit-weighting balanced \
  --max-call-rate 0.04 \
  --max-false-call-rate 0.02 \
  --min-oof-recall 0.10 \
  --thresholds 0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.925,0.95,0.975,0.99

echo "MUNO21 v11 Tool-Need gate complete: ${RUN_DIR}"
