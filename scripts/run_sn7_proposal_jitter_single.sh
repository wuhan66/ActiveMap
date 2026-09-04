#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 INPUT_MODE SEED PHYSICAL_GPU" >&2
  exit 2
fi

MODE="$1"
SEED="$2"
GPU="$3"
if [[ "${MODE}" != "image_prior" && "${MODE}" != "prior_only" ]]; then
  echo "unsupported input mode: ${MODE}" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
ENCODER="${STORAGE_ROOT}/models/change_mamba/vssm_tiny_0230_ckpt_epoch_262.pth"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
LOG_ROOT="${STORAGE_ROOT}/logs/changemamba_proposal_jitter_three_seed_20260726"
RUN="${RUN_ROOT}/proposal_jitter16_${MODE}_seed${SEED}_v1"

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -e "${RUN}" ]]; then
  echo "Refusing to overwrite proposal-jitter run: ${RUN}" >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/train_sn7_changemamba.py \
  "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${RUN}" \
  --encoder-checkpoint "${ENCODER}" \
  --device cuda:0 --image-size 128 --batch-size 16 --workers 8 \
  --epochs 40 --min-epochs 10 --patience 8 \
  --learning-rate 0.0001 --positive-class-weight 5 \
  --max-translation-pixels 16 --input-mode "${MODE}" --seed "${SEED}" \
  > "${LOG_ROOT}/${MODE}_seed${SEED}.log" 2>&1
