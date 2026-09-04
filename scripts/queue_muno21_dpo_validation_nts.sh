#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: queue_muno21_dpo_validation_nts.sh RUN_ROOT TRAIN_PID GPU LABEL}"
TRAIN_PID="${2:?usage: queue_muno21_dpo_validation_nts.sh RUN_ROOT TRAIN_PID GPU LABEL}"
GPU="${3:?usage: queue_muno21_dpo_validation_nts.sh RUN_ROOT TRAIN_PID GPU LABEL}"
LABEL="${4:?usage: queue_muno21_dpo_validation_nts.sh RUN_ROOT TRAIN_PID GPU LABEL}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
PYTHON="${PYTHON:-/home/wh/venvs/activemap/bin/python}"
LOG_DIR="/mnt/mydisk/wh/ActiveMap/logs"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}/evaluation"

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  echo "[$(date --iso-8601=seconds)] waiting for DPO PID ${TRAIN_PID}"
  sleep 60
done

for path in \
  "${RUN_ROOT}/final/adapter_model.safetensors" \
  "${RUN_ROOT}/train_metrics.json" \
  "${RUN_ROOT}/eval_metrics.json"; do
  [[ -s "${path}" ]] || {
    echo "DPO ended without required artifact: ${path}" >&2
    exit 2
  }
done

while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

cd "${PROJECT_ROOT}"
PROJECT_ROOT="${PROJECT_ROOT}" bash scripts/evaluate_muno21_agent_run_adapter.sh \
  "${RUN_ROOT}" "${RUN_ROOT}/final" "${LABEL}" "${GPU}"

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" scripts/summarize_agent_checkpoints.py \
  "${RUN_ROOT}/evaluation" "${RUN_ROOT}/evaluation/selection" \
  --labels "${LABEL}"

touch "${RUN_ROOT}/evaluation/COMPLETED"
echo "[$(date --iso-8601=seconds)] ${LABEL} validation completed"
