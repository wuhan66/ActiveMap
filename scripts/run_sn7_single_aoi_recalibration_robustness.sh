#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-opencd/bin/python"
CHANGE_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
BAN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
OUTPUT_ROOT="${STORAGE_ROOT}/runs/single_aoi_recalibration_20260727"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

seed_paths() {
  local root="$1"
  local split="$2"
  for seed in 20260725 20260726 20260727; do
    printf '%s\n' "${root}/full_weight5_seed${seed}_v1/${split}/per_sample.jsonl"
  done
}

mapfile -t change_train < <(seed_paths "${CHANGE_ROOT}" train_audit)
mapfile -t change_val < <(seed_paths "${CHANGE_ROOT}" val_audit)
mapfile -t ban_train < <(seed_paths "${BAN_ROOT}" train_audit)
mapfile -t ban_val < <(seed_paths "${BAN_ROOT}" val_audit)

"${PYTHON}" scripts/evaluate_sn7_single_aoi_recalibration_robustness.py \
  "${OUTPUT_ROOT}/changemamba_to_ban.json" \
  --source-train "${change_train[@]}" \
  --target-train "${ban_train[@]}" \
  --target-val "${ban_val[@]}" \
  --source-backend ChangeMamba --target-backend BAN \
  --output-markdown "${OUTPUT_ROOT}/changemamba_to_ban.md" \
  --output-figure "${OUTPUT_ROOT}/changemamba_to_ban.png" \
  >"${OUTPUT_ROOT}/changemamba_to_ban.log" 2>&1 &
pid1="$!"

"${PYTHON}" scripts/evaluate_sn7_single_aoi_recalibration_robustness.py \
  "${OUTPUT_ROOT}/ban_to_changemamba.json" \
  --source-train "${ban_train[@]}" \
  --target-train "${change_train[@]}" \
  --target-val "${change_val[@]}" \
  --source-backend BAN --target-backend ChangeMamba \
  --output-markdown "${OUTPUT_ROOT}/ban_to_changemamba.md" \
  --output-figure "${OUTPUT_ROOT}/ban_to_changemamba.png" \
  >"${OUTPUT_ROOT}/ban_to_changemamba.log" 2>&1 &
pid2="$!"

wait "${pid1}"
wait "${pid2}"
