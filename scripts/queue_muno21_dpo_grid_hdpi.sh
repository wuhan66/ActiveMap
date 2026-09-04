#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
ROOT="${STORAGE_ROOT}/runs/agent"
LOG_ROOT="${STORAGE_ROOT}/logs/muno21_dpo_beta005_010_formal_grid_20260802"
mkdir -p "${LOG_ROOT}"

launch_when_ready() {
  local gpu="$1" beta="$2" seed="$3"
  local run="${ROOT}/muno21_qwen3_4b_safety_dpo_beta${beta}_seed${seed}_v1"
  local label="dpo-beta${beta}-seed${seed}"
  local log="${LOG_ROOT}/${label}.log"

  while [[ ! -f "${run}/final/TRANSFER_COMPLETE" ]] || \
        [[ ! -s "${run}/final/adapter_model.safetensors" ]] || \
        [[ ! -s "${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v6_anonymized/val/sft.jsonl" ]] || \
        [[ ! -s "${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v6_anonymized/val/trajectories.jsonl" ]]; do
    sleep 20
  done
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    sleep 20
  done
  if [[ -f "${run}/evaluation/COMPLETED" ]]; then
    echo "${label} already complete"
    return 0
  fi
  PROJECT_ROOT="${PROJECT_ROOT}" STORAGE_ROOT="${STORAGE_ROOT}" \
    PYTHON="${PYTHON}" PEFT_OVERLAY="${STORAGE_ROOT}/envs/agent_peft_overlay" \
    MUNO21_UPDATER_CHECKPOINT="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt" \
    bash "${PROJECT_ROOT}/scripts/evaluate_muno21_agent_run_adapter.sh" \
      "${run}" "${run}/final" "${label}" "${gpu}"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" "${PYTHON}" \
    "${PROJECT_ROOT}/scripts/summarize_agent_checkpoints.py" \
    "${run}/evaluation" "${run}/evaluation/selection" --labels "${label}"
  date -Is >"${run}/evaluation/COMPLETED"
}

(launch_when_ready 1 005 20260822) >"${LOG_ROOT}/beta005_seed20260822.log" 2>&1 &
(launch_when_ready 2 005 20260823) >"${LOG_ROOT}/beta005_seed20260823.log" 2>&1 &
(launch_when_ready 3 010 20260822) >"${LOG_ROOT}/beta010_seed20260822.log" 2>&1 &
(launch_when_ready 4 010 20260823) >"${LOG_ROOT}/beta010_seed20260823.log" 2>&1 &

printf '%s\n' "$!" >"${LOG_ROOT}/launcher.pid"
wait
date -Is >"${LOG_ROOT}/GRID_COMPLETED"
