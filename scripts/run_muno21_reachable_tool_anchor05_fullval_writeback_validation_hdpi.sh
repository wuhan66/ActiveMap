#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
UPDATER="${UPDATER:-${STORE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
EPISODES="${EPISODES:-${STORE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
ROOT="${ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_anchor05_fullval_validation_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORE}/runs/agent/muno21_reachable_tool_support_anchor05_fullval_writeback_v1}"
LOGROOT="${LOGROOT:-${STORE}/logs/muno21_reachable_tool_support_anchor05_fullval_writeback_v1}"

for path in "${PY}" "${UPDATER}" "${EPISODES}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done

mkdir -p "${OUTPUT_ROOT}" "${LOGROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

launch_one() {
  local gpu="$1"
  local variant="$2"
  local seed="$3"
  local rollout="${ROOT}/${variant}_seed${seed}/qwen3_4b_sft_tool_to_belief.jsonl"
  local output="${OUTPUT_ROOT}/${variant}_seed${seed}"
  local log="${LOGROOT}/${variant}_seed${seed}.log"
  if [[ -s "${output}/summary.json" && -s "${output}/writeback.jsonl" ]]; then
    echo "full-val writeback already complete: ${variant} seed${seed}"
    return 0
  fi
  [[ -s "${rollout}" ]] || { echo "Missing rollout: ${rollout}" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES="${gpu}" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    nohup "${PY}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${rollout}" "${output}" \
    --device cuda:0 --split val --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --protocol-name "muno21-anchor05-real-writeback-fullval-v1" \
    >"${log}" 2>&1 &
  echo "$! GPU${gpu} full-val writeback ${variant} seed${seed}"
}

# GPU0/6 are intentionally excluded.
launch_one 1 anchor05 20261230
launch_one 2 sft_control 20261230
launch_one 3 anchor05 20261232
launch_one 4 sft_control 20261232
launch_one 5 anchor05 20261233
launch_one 7 sft_control 20261233
wait

for variant in anchor05 sft_control; do
  for seed in 20261230 20261232 20261233; do
    output="${OUTPUT_ROOT}/${variant}_seed${seed}"
    [[ -s "${output}/summary.json" && -s "${output}/writeback.jsonl" ]] || {
      echo "full-val writeback did not complete: ${variant} seed${seed}" >&2
      exit 4
    }
  done
done

echo "anchor05 full-val writebacks completed"
