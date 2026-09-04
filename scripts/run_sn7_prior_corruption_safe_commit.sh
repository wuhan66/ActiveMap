#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
MODEL_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
CORRUPTION_ROOT="${MODEL_ROOT}/prior_input_corruption_v1"
OUTPUT_ROOT="${MODEL_ROOT}/prior_corruption_safe_commit_v1"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_prior_corruption_safe_commit_v1"
GPUS=(
  "${SAFE_COMMIT_GPU0:-0}"
  "${SAFE_COMMIT_GPU1:-1}"
  "${SAFE_COMMIT_GPU2:-2}"
)
MODEL_SEEDS=(20260725 20260726 20260727)
SEVERITIES=(0 4 8 16)

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_train_predictions() {
  local gpu="$1"
  local seed="$2"
  local checkpoint="${MODEL_ROOT}/proposal_jitter16_image_prior_seed${seed}_v1/best.pt"
  local output="${OUTPUT_ROOT}/train_predictions_seed${seed}"
  if [[ -s "${output}/summary.json" ]]; then
    return
  fi
  if [[ -e "${output}" ]]; then
    echo "Refusing partial train prediction output: ${output}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/evaluate_sn7_changemamba.py \
    "${checkpoint}" "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${output}" \
    --split train --device cuda:0 --batch-size 16 --workers 8 \
    --input-mode image_prior --no-save-masks \
    > "${LOG_ROOT}/train_predictions_seed${seed}.log" 2>&1
}

pids=()
for index in "${!MODEL_SEEDS[@]}"; do
  run_train_predictions "${GPUS[$index]}" "${MODEL_SEEDS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

for seed in "${MODEL_SEEDS[@]}"; do
  calibration="${OUTPUT_ROOT}/calibration_seed${seed}"
  if [[ ! -s "${calibration}/summary.json" ]]; then
    if [[ -e "${calibration}" ]]; then
      echo "Refusing partial calibration output: ${calibration}" >&2
      exit 1
    fi
    "${PYTHON}" scripts/calibrate_sn7_changemamba_safe_commit.py \
      "${OUTPUT_ROOT}/train_predictions_seed${seed}/per_sample.jsonl" \
      "${CORRUPTION_ROOT}/severity0_model_seed${seed}/per_sample.jsonl" \
      "${calibration}" --folds 5 --l2 0.01 \
      > "${LOG_ROOT}/calibration_seed${seed}.log" 2>&1
  fi
  for severity in "${SEVERITIES[@]}"; do
    output="${OUTPUT_ROOT}/severity${severity}_seed${seed}"
    if [[ -s "${output}/summary.json" ]]; then
      continue
    fi
    if [[ -e "${output}" ]]; then
      echo "Refusing partial frozen application output: ${output}" >&2
      exit 1
    fi
    "${PYTHON}" scripts/apply_sn7_changemamba_safe_commit.py \
      "${calibration}/summary.json" \
      "${CORRUPTION_ROOT}/severity${severity}_model_seed${seed}/per_sample.jsonl" \
      "${output}" > "${LOG_ROOT}/severity${severity}_seed${seed}.log" 2>&1
  done
done

"${PYTHON}" scripts/summarize_sn7_prior_corruption_safe_commit.py \
  "${OUTPUT_ROOT}" --bootstrap-repetitions 5000 --bootstrap-seed 20260730 \
  > "${OUTPUT_ROOT}/summarizer.log" 2>&1
"${PYTHON}" scripts/plot_sn7_prior_corruption_safe_commit.py \
  "${OUTPUT_ROOT}/summary.json" \
  "${OUTPUT_ROOT}/frozen_safe_commit_corruption_curve"
