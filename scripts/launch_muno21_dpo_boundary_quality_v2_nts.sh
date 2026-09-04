#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${PYTHON:-/home/wh/venvs/activemap/bin/python}"
LOG_ROOT="${STORAGE_ROOT}/logs/muno21_dpo_boundary_quality_v2_20260802"
mkdir -p "${LOG_ROOT}"

run_one() {
  local gpu="$1" beta="$2"
  local run="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_safety_dpo_beta${beta}_seed20260821_v1"
  local label="dpo-beta${beta}-seed20260821-quality-v2"
  [[ -s "${run}/final/adapter_model.safetensors" ]]
  [[ ! -e "${run}/evaluation/${label}" ]] || {
    echo "refusing existing output: ${run}/evaluation/${label}" >&2
    return 3
  }
  PROJECT_ROOT="${PROJECT_ROOT}" bash "${PROJECT_ROOT}/scripts/evaluate_muno21_agent_run_adapter.sh" \
    "${run}" "${run}/final" "${label}" "${gpu}"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" "${PYTHON}" \
    "${PROJECT_ROOT}/scripts/summarize_agent_checkpoints.py" \
    "${run}/evaluation" "${run}/evaluation/selection_quality_v2" --labels "${label}"
  date -Is >"${run}/evaluation/QUALITY_V2_COMPLETED"
}

(run_one 3 020) >"${LOG_ROOT}/beta020_seed20260821.log" 2>&1 &
(run_one 0 050) >"${LOG_ROOT}/beta050_seed20260821.log" 2>&1 &
wait
date -Is >"${LOG_ROOT}/BOUNDARY_COMPLETED"
