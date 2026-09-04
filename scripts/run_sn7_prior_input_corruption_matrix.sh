#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
SOURCE_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
OUTPUT_ROOT="${SOURCE_ROOT}/prior_input_corruption_v1"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_prior_input_corruption_v1"
POLL_SECONDS="${POLL_SECONDS:-60}"
GPUS=(
  "${PRIOR_CORRUPTION_GPU0:-0}"
  "${PRIOR_CORRUPTION_GPU1:-1}"
  "${PRIOR_CORRUPTION_GPU2:-2}"
  "${PRIOR_CORRUPTION_GPU3:-3}"
)
SEVERITIES=(0 4 8 16)
MODEL_SEEDS=(20260725 20260726 20260727)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

gpu_is_idle() {
  local gpu="$1"
  local memory utilization
  IFS=, read -r memory utilization < <(
    nvidia-smi --id="${gpu}" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits
  )
  memory="${memory// /}"
  utilization="${utilization// /}"
  [[ "${memory}" -lt 1000 && "${utilization}" -lt 15 ]]
}

wait_for_gpu() {
  local gpu="$1"
  until gpu_is_idle "${gpu}"; do
    echo "Waiting for GPU ${gpu}."
    sleep "${POLL_SECONDS}"
  done
}

run_severity() {
  local gpu="$1"
  local severity="$2"
  wait_for_gpu "${gpu}"
  for model_seed in "${MODEL_SEEDS[@]}"; do
    local checkpoint="${SOURCE_ROOT}/proposal_jitter16_image_prior_seed${model_seed}_v1/best.pt"
    local output="${OUTPUT_ROOT}/severity${severity}_model_seed${model_seed}"
    local log="${LOG_ROOT}/severity${severity}_model_seed${model_seed}.log"
    if [[ -s "${output}/summary.json" ]]; then
      continue
    fi
    if [[ -e "${output}" ]]; then
      echo "Refusing partial output: ${output}" >&2
      return 1
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
      scripts/evaluate_sn7_changemamba.py \
      "${checkpoint}" "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${output}" \
      --split val --device cuda:0 --batch-size 16 --workers 8 \
      --input-mode image_prior --no-save-masks \
      --prior-input-translation-pixels "${severity}" \
      --corruption-seed "${model_seed}" \
      > "${log}" 2>&1
  done
}

pids=()
for index in "${!SEVERITIES[@]}"; do
  run_severity "${GPUS[$index]}" "${SEVERITIES[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
