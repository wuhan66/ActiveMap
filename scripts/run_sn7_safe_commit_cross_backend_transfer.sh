#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-opencd/bin/python"
CHANGE_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
BAN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
OUTPUT_ROOT="${STORAGE_ROOT}/runs/safe_commit_cross_backend_20260727"
LOG_ROOT="${STORAGE_ROOT}/logs/safe_commit_cross_backend_20260727"

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_transfer() {
  local source="$1"
  local target="$2"
  local seed="$3"
  local source_root="$4"
  local target_root="$5"
  local output="${OUTPUT_ROOT}/${source}_to_${target}_seed${seed}"
  "${PYTHON}" scripts/evaluate_sn7_safe_commit_transfer.py \
    "${source_root}/full_weight5_seed${seed}_v1/train_audit/per_sample.jsonl" \
    "${target_root}/full_weight5_seed${seed}_v1/val_audit/per_sample.jsonl" \
    "${output}" \
    --source-backend "${source}" --target-backend "${target}" \
    --folds 5 --l2 0.01 \
    >"${LOG_ROOT}/${source}_to_${target}_seed${seed}.log" 2>&1
}

pids=()
for seed in 20260725 20260726 20260727; do
  run_transfer changemamba ban "${seed}" "${CHANGE_ROOT}" "${BAN_ROOT}" &
  pids+=("$!")
  run_transfer ban changemamba "${seed}" "${BAN_ROOT}" "${CHANGE_ROOT}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

for direction in changemamba_to_ban ban_to_changemamba; do
  runs=()
  for seed in 20260725 20260726 20260727; do
    runs+=("${OUTPUT_ROOT}/${direction}_seed${seed}")
  done
  "${PYTHON}" scripts/aggregate_sn7_changemamba_safe_commit.py \
    "${OUTPUT_ROOT}/${direction}_three_seed.json" "${runs[@]}" \
    --output-markdown "${OUTPUT_ROOT}/${direction}_three_seed.md"
  "${PYTHON}" scripts/bootstrap_sn7_changemamba_safe_commit.py \
    "${OUTPUT_ROOT}/${direction}_aoi_bootstrap.json" "${runs[@]}" \
    --draws 5000 --seed 20260727 \
    --output-markdown "${OUTPUT_ROOT}/${direction}_aoi_bootstrap.md"
done
