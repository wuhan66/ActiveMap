#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORE}/envs/activemap-agent/bin/python}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_rollouts_v1}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_reachable_tool_support_writebacks_v1}"

for path in "${PYTHON}" "${UPDATER}" "${EPISODES}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

mkdir -p "${LOGROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

launch_one() {
  local gpu="$1"
  local index="$2"
  local seed="$3"
  local run="${ROOT}/rollout${index}_seed${seed}"
  local rollout="${run}/qwen3_4b_sft_tool_to_belief.jsonl"
  local output="${run}/writeback"
  local log="${LOGROOT}/rollout${index}_seed${seed}.log"
  if [[ -s "${output}/summary.json" ]]; then
    echo "writeback already complete: ${output}"
    return 0
  fi
  [[ -s "${rollout}" ]] || { echo "Missing rollout: ${rollout}" >&2; return 1; }
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON}" \
    scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${rollout}" "${output}" \
    --device cuda:0 --split train --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    >"${log}" 2>&1 &
  echo "$! GPU${gpu} writeback${index}"
}

launch_one 1 0 20261121
launch_one 2 1 20261122
launch_one 3 2 20261123
launch_one 4 3 20261124
wait

for index in 0 1 2 3; do
  seed=$((20261121 + index))
  [[ -s "${ROOT}/rollout${index}_seed${seed}/writeback/writeback.jsonl" ]] || {
    echo "Missing completed writeback for rollout${index}" >&2
    exit 3
  }
done

echo "recurrent executable writeback preparation complete"
