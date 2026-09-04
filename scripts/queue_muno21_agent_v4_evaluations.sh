#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-2}"
PROJECT_ROOT="/home/wh/projects/activemap-v1"
RUN_ROOT="/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v4_no_acquire_oversample_seed20260821"
TRAIN_PID_FILE="${RUN_ROOT}/train.pid"
STATUS_PATH="${RUN_ROOT}/evaluation/queue.exit_code"
LABELS=(checkpoint-200 checkpoint-300 checkpoint-352)

mkdir -p "${RUN_ROOT}/evaluation"
rm -f "${STATUS_PATH}"
trap 'status=$?; printf "%s\n" "$status" > "${STATUS_PATH}"' EXIT

gpu_is_free() {
  [[ -z "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]
}

training_is_running() {
  [[ -s "${TRAIN_PID_FILE}" ]] && kill -0 "$(cat "${TRAIN_PID_FILE}")" 2>/dev/null
}

wait_for_gpu() {
  while ! gpu_is_free; do
    echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}"
    sleep 30
  done
}

evaluate_when_ready() {
  local label="$1"
  local adapter="${RUN_ROOT}/checkpoints/${label}"
  while [[ ! -s "${adapter}/adapter_model.safetensors" ]]; do
    if ! training_is_running; then
      echo "[$(date --iso-8601=seconds)] missing ${label} after training exit" >&2
      return 1
    fi
    echo "[$(date --iso-8601=seconds)] waiting for ${label}"
    sleep 30
  done
  wait_for_gpu
  "${PROJECT_ROOT}/scripts/evaluate_muno21_agent_run_adapter.sh" \
    "${RUN_ROOT}" "${adapter}" "${label}" "${GPU}"
}

for label in "${LABELS[@]}"; do
  evaluate_when_ready "${label}"
done

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  /home/wh/venvs/activemap/bin/python \
  "${PROJECT_ROOT}/scripts/summarize_agent_checkpoints.py" \
  "${RUN_ROOT}/evaluation" "${RUN_ROOT}/evaluation/selection" \
  --labels checkpoint-200,checkpoint-300,checkpoint-352
