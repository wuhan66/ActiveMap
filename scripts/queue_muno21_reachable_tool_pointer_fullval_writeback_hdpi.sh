#!/usr/bin/env bash
set -euo pipefail

# Evaluate the final greedy pointer policy by executable map quality on the
# complete validation rollout. This is a diagnostic and never reads test data.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_pointer_final_fullval_v3}"
ROLLOUT="${ROLLOUT:-${ROOT}/qwen3_4b_sft_tool_to_belief.jsonl}"
OUTPUT="${OUTPUT:-${ROOT}/writeback}"
LOG="${LOG:-${STORE}/logs/muno21_reachable_tool_pointer_final_fullval_v3.writeback.log}"
GPU="${GPU:-7}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT}"
mkdir -p "$(dirname "${LOG}")"
[[ ! -e "${OUTPUT}" ]] || {
  echo "Refusing existing output: ${OUTPUT}" >&2
  exit 2
}
for path in "${PY}" "${UPDATER}" "${EPISODES}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done

until [[ -s "${ROLLOUT}" ]]; do
  echo "Waiting for full validation rollout: ${ROLLOUT}"
  sleep "${POLL_SECONDS}"
done

export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" scripts/evaluate_agent_map_writeback.py \
  "${UPDATER}" "${EPISODES}" "${ROLLOUT}" "${OUTPUT}" \
  --device cuda:0 --split val --image-size 512 --threshold 0.5 \
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
  >"${LOG}" 2>&1

echo "Pointer full-validation writeback completed: ${OUTPUT}"
