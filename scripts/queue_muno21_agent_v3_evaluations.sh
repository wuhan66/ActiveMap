#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-2}"
RUN_ROOT="/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821"
TRAIN_PID_FILE="${RUN_ROOT}/train.pid"
PROJECT_ROOT="/home/wh/projects/activemap-v1"

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

run_corrected_selector_rollout() {
  local output="/mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_ensemble_v6_corrected_rollout"
  if [[ -s "${output}/rollout/summary.json" ]]; then
    echo "[$(date --iso-8601=seconds)] corrected selector rollout already exists"
    return 0
  fi
  wait_for_gpu
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${GPU}" \
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    /home/wh/venvs/activemap/bin/python \
    "${PROJECT_ROOT}/scripts/evaluate_evidence_selector_ensemble.py" \
    /mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/selector_states_v1.jsonl \
    "${output}/one_step.csv" \
    --checkpoint /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260811/best.pt \
    --checkpoint /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260812/best.pt \
    --checkpoint /mnt/mydisk/wh/ActiveMap/runs/selector/muno21_evidence_conservative_v5_seed20260813/best.pt \
    --split val --budgets 1.5,3.0,4.5 --device cuda \
    --rollout-output "${output}/rollout"
}

evaluate_when_ready() {
  local adapter="$1"
  local label="$2"
  local model_file="${adapter}/adapter_model.safetensors"
  while [[ ! -s "${model_file}" ]]; do
    if ! training_is_running; then
      echo "[$(date --iso-8601=seconds)] skip ${label}: training ended without adapter"
      return 0
    fi
    echo "[$(date --iso-8601=seconds)] waiting for ${label}"
    sleep 30
  done
  wait_for_gpu
  "${PROJECT_ROOT}/scripts/evaluate_muno21_agent_adapter.sh" \
    "${adapter}" "${label}" "${GPU}"
}

run_corrected_selector_rollout
evaluate_when_ready "${RUN_ROOT}/checkpoints/checkpoint-200" "checkpoint-200"
evaluate_when_ready "${RUN_ROOT}/checkpoints/checkpoint-400" "checkpoint-400"
evaluate_when_ready "${RUN_ROOT}/checkpoints/checkpoint-418" "checkpoint-418"

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  /home/wh/venvs/activemap/bin/python \
  "${PROJECT_ROOT}/scripts/summarize_agent_checkpoints.py" \
  "${RUN_ROOT}/evaluation" "${RUN_ROOT}/evaluation/selection" \
  --labels checkpoint-200,checkpoint-400,checkpoint-418
