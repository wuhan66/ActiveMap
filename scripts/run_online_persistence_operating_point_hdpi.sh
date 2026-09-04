#!/usr/bin/env bash
set -euo pipefail

# Validation-only operating-point audit for true online persistence. The 0.70
# point was evaluated during the original chronological-training sweep.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/reviewer_closure_20260814/online_persistence_operating_point_v1}"
VAL_EPISODES="${VAL_EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl}"
ASSET_MAP="/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
SEEDS=(20260716 20260719 20260722)
CHECKPOINTS=(
  "${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
  "${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260719/best_quality.pt"
  "${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260722/best_quality.pt"
)

# Sensitivity audit only. This does not replace the declared 80% false-edit
# constraint in online_persistence_matrix_v2.
CONFIDENCE_THRESHOLD="${CONFIDENCE_THRESHOLD:-0.70}"
REPLAY_IOU_THRESHOLD="${REPLAY_IOU_THRESHOLD:-0.99}"

test -f "${VAL_EPISODES}"
test ! -e "${RUN_ROOT}"
for checkpoint in "${CHECKPOINTS[@]}"; do test -f "${checkpoint}"; done
mkdir -p "${RUN_ROOT}/val" "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

declare -a pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  checkpoint="${CHECKPOINTS[$index]}"
  CUDA_VISIBLE_DEVICES="${index}" nohup "${PYTHON}" scripts/evaluate_online_persistent_map_maintenance.py \
    "${VAL_EPISODES}" "${checkpoint}" "${RUN_ROOT}/val/seed${seed}" \
    --split val --device cuda:0 --image-size 512 --minimum-chain-length 2 \
    --safe-confidence-threshold "${CONFIDENCE_THRESHOLD}" \
    --safe-replay-iou-threshold "${REPLAY_IOU_THRESHOLD}" \
    --asset-root-map "${ASSET_MAP}" \
    >"${RUN_ROOT}/logs/val_seed${seed}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || exit "${status}"

"${PYTHON}" scripts/aggregate_online_persistent_map_maintenance.py \
  "${RUN_ROOT}/three_seed_validation_summary.json" \
  --record "20260716=${RUN_ROOT}/val/seed20260716/online_persistent_traces.jsonl" \
  --record "20260719=${RUN_ROOT}/val/seed20260719/online_persistent_traces.jsonl" \
  --record "20260722=${RUN_ROOT}/val/seed20260722/online_persistent_traces.jsonl" \
  --bootstrap-repetitions 10000 --seed 20260814 \
  >"${RUN_ROOT}/logs/aggregate_validation.log" 2>&1

echo "completed online persistence operating point: ${RUN_ROOT}"
