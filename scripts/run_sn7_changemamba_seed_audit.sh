#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 SEED PHYSICAL_GPU" >&2
  exit 2
fi

SEED="$1"
GPU="$2"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python}"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
RUN="${RUN_ROOT}/full_weight5_seed${SEED}_v1"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CHANGE_CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ ! -s "${RUN}/summary.json" || ! -s "${RUN}/best.pt" ]]; then
  echo "seed ${SEED} has not completed training" >&2
  exit 1
fi

evaluate_split() {
  local split="$1"
  local output="${RUN}/${split}_audit"
  if [[ -s "${output}/summary.json" ]]; then
    return
  fi
  if [[ -e "${output}" ]]; then
    echo "refusing partial audit directory: ${output}" >&2
    exit 1
  fi
  local extra=()
  if [[ "${split}" == "train" ]]; then
    extra+=(--no-save-masks)
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_sn7_changemamba.py \
    "${RUN}/best.pt" \
    "${MANIFEST}" \
    "${CHANGE_REPO}" \
    "${CHANGE_CONFIG}" \
    "${output}" \
    --split "${split}" \
    --device cuda:0 \
    --batch-size 16 \
    --workers 8 \
    --image-size 128 \
    "${extra[@]}"
}

evaluate_split train
evaluate_split val

SAFE="${RUN}/safe_commit"
if [[ ! -s "${SAFE}/summary.json" ]]; then
  if [[ -e "${SAFE}" ]]; then
    echo "refusing partial safe-commit directory: ${SAFE}" >&2
    exit 1
  fi
  "${PYTHON}" scripts/calibrate_sn7_changemamba_safe_commit.py \
    "${RUN}/train_audit/per_sample.jsonl" \
    "${RUN}/val_audit/per_sample.jsonl" \
    "${SAFE}" \
    --folds 5 \
    --l2 0.01
fi
