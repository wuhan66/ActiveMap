#!/usr/bin/env bash
set -euo pipefail

# A real online persistence audit: all online branches re-run perception using
# carried committed geometry, while each Safe Commit threshold is selected on
# chronological training chains only.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/reviewer_closure_20260814/online_persistence_matrix_v2}"
TRAIN_EPISODES="${TRAIN_EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl}"
VAL_EPISODES="${VAL_EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl}"
ASSET_MAP="/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
SEEDS=(20260716 20260719 20260722)
CHECKPOINTS=(
  "${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
  "${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260719/best_quality.pt"
  "${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260722/best_quality.pt"
)
THRESHOLDS=(0.40 0.55 0.70 0.85)

test -f "${TRAIN_EPISODES}"
test -f "${VAL_EPISODES}"
test ! -e "${RUN_ROOT}"
for checkpoint in "${CHECKPOINTS[@]}"; do test -f "${checkpoint}"; done
mkdir -p "${RUN_ROOT}/train" "${RUN_ROOT}/val" "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

declare -a pids=()
declare -a labels=()
wait_for_slot() {
  while (( ${#pids[@]} >= 4 )); do
    wait "${pids[0]}" || { echo "failed: ${labels[0]}" >&2; exit 1; }
    pids=("${pids[@]:1}")
    labels=("${labels[@]:1}")
  done
}
job=0
for index in "${!SEEDS[@]}"; do
  for threshold in "${THRESHOLDS[@]}"; do
    seed="${SEEDS[$index]}"
    checkpoint="${CHECKPOINTS[$index]}"
    key="t${threshold/./p}"
    gpu=$((job % 4))
    wait_for_slot
    CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON}" scripts/evaluate_online_persistent_map_maintenance.py \
      "${TRAIN_EPISODES}" "${checkpoint}" "${RUN_ROOT}/train/seed${seed}_${key}" \
      --split train --device cuda:0 --image-size 512 --minimum-chain-length 2 \
      --safe-confidence-threshold "${threshold}" --safe-replay-iou-threshold 0.99 \
      --asset-root-map "${ASSET_MAP}" \
      >"${RUN_ROOT}/logs/train_seed${seed}_${key}.log" 2>&1 &
    pids+=("$!")
    labels+=("train seed=${seed} threshold=${threshold} gpu=${gpu}")
    ((job += 1))
  done
done
status=0
for index in "${!pids[@]}"; do
  wait "${pids[$index]}" || { echo "failed: ${labels[$index]}" >&2; status=1; }
done
(( status == 0 )) || exit "${status}"

for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  candidates=()
  for threshold in "${THRESHOLDS[@]}"; do
    key="t${threshold/./p}"
    candidates+=(--candidate "${threshold}=${RUN_ROOT}/train/seed${seed}_${key}/summary.json")
  done
  "${PYTHON}" scripts/select_online_persistence_gate.py \
    "${RUN_ROOT}/gate_seed${seed}.json" "${candidates[@]}" --false-edit-fraction 0.80 \
    >"${RUN_ROOT}/logs/select_gate_seed${seed}.log" 2>&1
done

pids=()
labels=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  checkpoint="${CHECKPOINTS[$index]}"
  threshold="$("${PYTHON}" -c "import json; print(json.load(open('${RUN_ROOT}/gate_seed${seed}.json'))['selected']['threshold'])")"
  gpu="${index}"
  CUDA_VISIBLE_DEVICES="${gpu}" nohup "${PYTHON}" scripts/evaluate_online_persistent_map_maintenance.py \
    "${VAL_EPISODES}" "${checkpoint}" "${RUN_ROOT}/val/seed${seed}" \
    --split val --device cuda:0 --image-size 512 --minimum-chain-length 2 \
    --safe-confidence-threshold "${threshold}" --safe-replay-iou-threshold 0.99 \
    --asset-root-map "${ASSET_MAP}" \
    >"${RUN_ROOT}/logs/val_seed${seed}.log" 2>&1 &
  pids+=("$!")
  labels+=("val seed=${seed} threshold=${threshold} gpu=${gpu}")
done
status=0
for index in "${!pids[@]}"; do
  wait "${pids[$index]}" || { echo "failed: ${labels[$index]}" >&2; status=1; }
done
(( status == 0 )) || exit "${status}"

"${PYTHON}" scripts/aggregate_online_persistent_map_maintenance.py \
  "${RUN_ROOT}/three_seed_validation_summary.json" \
  --record "20260716=${RUN_ROOT}/val/seed20260716/online_persistent_traces.jsonl" \
  --record "20260719=${RUN_ROOT}/val/seed20260719/online_persistent_traces.jsonl" \
  --record "20260722=${RUN_ROOT}/val/seed20260722/online_persistent_traces.jsonl" \
  --bootstrap-repetitions 10000 --seed 20260814 \
  >"${RUN_ROOT}/logs/aggregate_validation.log" 2>&1

echo "completed online persistence matrix: ${RUN_ROOT}"
