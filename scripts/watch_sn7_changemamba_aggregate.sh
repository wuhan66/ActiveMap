#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python}"
PAPER_PYTHON="${PAPER_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CHANGE_CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
FROZEN_MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
OUTPUT_JSON="${RUN_ROOT}/full_weight5_three_seed_20260726.json"
OUTPUT_MD="${RUN_ROOT}/full_weight5_three_seed_20260726.md"
declare -a runs=(
  "${RUN_ROOT}/full_weight5_seed20260725_v1"
  "${RUN_ROOT}/full_weight5_seed20260726_v1"
  "${RUN_ROOT}/full_weight5_seed20260727_v1"
)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
while true; do
  complete=1
  for run in "${runs[@]}"; do
    if [[ ! -s "${run}/summary.json" ]]; then
      complete=0
    fi
  done
  if [[ "${complete}" -eq 1 ]]; then
    break
  fi
  if ! pgrep -f \
    'launch_sn7_changemamba_after_migration|train_sn7_changemamba.py' \
    > /dev/null; then
    echo "Training pipeline ended before all three summaries were written." >&2
    exit 1
  fi
  sleep 120
done

for run in "${runs[@]}"; do
  train_audit="${run}/train_audit"
  audit="${run}/val_audit"
  if [[ ! -s "${train_audit}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/evaluate_sn7_changemamba.py \
      "${run}/best.pt" \
      "${FROZEN_MANIFEST}" \
      "${CHANGE_REPO}" \
      "${CHANGE_CONFIG}" \
      "${train_audit}" \
      --split train \
      --device cuda:0 \
      --batch-size 16 \
      --workers 8 \
      --image-size 128 \
      --no-save-masks
  fi
  if [[ ! -s "${audit}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/evaluate_sn7_changemamba.py \
      "${run}/best.pt" \
      "${FROZEN_MANIFEST}" \
      "${CHANGE_REPO}" \
      "${CHANGE_CONFIG}" \
      "${audit}" \
      --split val \
      --device cuda:0 \
      --batch-size 16 \
      --workers 8 \
      --image-size 128
  fi
  safe_commit="${run}/safe_commit"
  if [[ ! -s "${safe_commit}/summary.json" ]]; then
    "${PYTHON}" scripts/calibrate_sn7_changemamba_safe_commit.py \
      "${train_audit}/per_sample.jsonl" \
      "${audit}/per_sample.jsonl" \
      "${safe_commit}" \
      --folds 5 \
      --l2 0.01
  fi
done

"${PYTHON}" scripts/aggregate_sn7_changemamba.py \
  "${OUTPUT_JSON}" \
  "${runs[@]}" \
  --output-markdown "${OUTPUT_MD}"

"${PYTHON}" scripts/bootstrap_sn7_changemamba_audit.py \
  "${RUN_ROOT}/full_weight5_three_seed_aoi_bootstrap_20260726.json" \
  "${runs[0]}/val_audit" \
  "${runs[1]}/val_audit" \
  "${runs[2]}/val_audit" \
  --draws 5000 \
  --seed 20260726

"${PYTHON}" scripts/aggregate_sn7_changemamba_safe_commit.py \
  "${RUN_ROOT}/full_weight5_three_seed_safe_commit_20260726.json" \
  "${runs[0]}/safe_commit" \
  "${runs[1]}/safe_commit" \
  "${runs[2]}/safe_commit" \
  --output-markdown \
  "${RUN_ROOT}/full_weight5_three_seed_safe_commit_20260726.md"

"${PYTHON}" scripts/bootstrap_sn7_changemamba_safe_commit.py \
  "${RUN_ROOT}/full_weight5_three_seed_safe_commit_aoi_bootstrap_20260726.json" \
  "${runs[0]}/safe_commit" \
  "${runs[1]}/safe_commit" \
  "${runs[2]}/safe_commit" \
  --draws 5000 \
  --seed 20260726 \
  --output-markdown \
  "${RUN_ROOT}/full_weight5_three_seed_safe_commit_aoi_bootstrap_20260726.md"

BUNDLE_ROOT="${STORAGE_ROOT}/artifacts/paper_results/bundles"
mkdir -p "${BUNDLE_ROOT}"
ALWAYS_BUNDLE="${BUNDLE_ROOT}/changemamba_always_commit_val.json"
SAFE_BUNDLE="${BUNDLE_ROOT}/changemamba_safe_commit_val.json"
if [[ ! -s "${ALWAYS_BUNDLE}" ]]; then
  PYTHONPATH=.:src "${PAPER_PYTHON}" scripts/build_paper_updater_bundle.py \
    configs/experiments/paper_registry.yaml \
    "${ALWAYS_BUNDLE}" \
    --experiment-id sn7_changemamba_commit_policy \
    --variant always_commit \
    --seed-predictions "20260725=${runs[0]}/val_audit/predictions.jsonl" \
    --seed-predictions "20260726=${runs[1]}/val_audit/predictions.jsonl" \
    --seed-predictions "20260727=${runs[2]}/val_audit/predictions.jsonl"
fi
if [[ ! -s "${SAFE_BUNDLE}" ]]; then
  PYTHONPATH=.:src "${PAPER_PYTHON}" scripts/build_paper_updater_bundle.py \
    configs/experiments/paper_registry.yaml \
    "${SAFE_BUNDLE}" \
    --experiment-id sn7_changemamba_commit_policy \
    --variant safe_commit \
    --seed-predictions "20260725=${runs[0]}/safe_commit/predictions.jsonl" \
    --seed-predictions "20260726=${runs[1]}/safe_commit/predictions.jsonl" \
    --seed-predictions "20260727=${runs[2]}/safe_commit/predictions.jsonl"
fi

FINALIZATION="${RUN_ROOT}/full_weight5_validation_finalization_20260726.json"
if [[ ! -s "${FINALIZATION}" ]]; then
  "${PYTHON}" scripts/finalize_sn7_changemamba_validation.py \
    "${RUN_ROOT}" \
    "${BUNDLE_ROOT}" \
    "${FINALIZATION}" \
    --seed 20260725 \
    --seed 20260726 \
    --seed 20260727
fi
