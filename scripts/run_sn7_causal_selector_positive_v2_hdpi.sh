#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${GPU:-3}"
SAMPLES="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_selector_runtime_grid_budget3_causal_v1/selector_states_train_val_causal_v1.jsonl"
CONFIG="${PROJECT_ROOT}/configs/selector/sn7_causal_online_runtime_grid_positive_v2.yaml"
RUN="${STORAGE_ROOT}/runs/selector/sn7_causal_online_runtime_grid_positive_v2_seed20260902"

[[ -x "${PYTHON}" && -f "${SAMPLES}" && -f "${CONFIG}" ]] || {
  echo "missing Python, causal samples, or positive-aware config" >&2
  exit 2
}
[[ ! -e "${RUN}" ]] || {
  echo "positive-aware run already exists: ${RUN}" >&2
  exit 2
}

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

"${PYTHON}" -m activemap.cli train-selector "${CONFIG}" --output "${RUN}"
"${PYTHON}" scripts/diagnose_selector_stop_frontier.py \
  "${RUN}/best.pt" "${SAMPLES}" "${RUN}/validation_stop_frontier.json" \
  --split val --device cuda:0 --batch-size 512 --points 201 \
  --max-false-call-rate 0.02 \
  --max-harmful-call-fraction 0.20 \
  --min-acquire-recall 0.10

echo "completed positive-aware causal selector under ${RUN}"
