#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
DATA_ROOT="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20"
MANIFEST="${DATA_ROOT}/train_val_complete_20260726.jsonl"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
ENCODER="${STORAGE_ROOT}/models/change_mamba/vssm_tiny_0230_ckpt_epoch_262.pth"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
LOG_ROOT="${STORAGE_ROOT}/logs/changemamba_modality_20260726"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-120}"

mkdir -p "${LOG_ROOT}"
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
    echo "GPU ${gpu} is occupied; waiting ${GPU_WAIT_SECONDS}s."
    sleep "${GPU_WAIT_SECONDS}"
  done
}

run_one() {
  local gpu="$1"
  local mode="$2"
  local seed="$3"
  local output="${RUN_ROOT}/full_weight5_${mode}_seed${seed}_v1"
  local log="${LOG_ROOT}/${mode}_seed${seed}.log"
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing run: ${output}" >&2
    return 1
  fi
  wait_for_gpu "${gpu}"
  echo "Launching ${mode} seed ${seed} on physical GPU ${gpu}."
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_sn7_changemamba.py \
    "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${output}" \
    --encoder-checkpoint "${ENCODER}" \
    --device cuda:0 \
    --image-size 128 \
    --batch-size 16 \
    --workers 8 \
    --epochs 40 \
    --min-epochs 10 \
    --patience 8 \
    --learning-rate 0.0001 \
    --positive-class-weight 5 \
    --input-mode "${mode}" \
    --seed "${seed}" \
    > "${log}" 2>&1
}

run_worker() {
  local gpu="$1"
  shift
  while [[ "$#" -gt 0 ]]; do
    run_one "${gpu}" "$1" "$2"
    shift 2
  done
}

run_worker 1 image_only 20260725 prior_only 20260726 &
pid1="$!"
run_worker 2 image_only 20260726 prior_only 20260727 &
pid2="$!"
run_worker 3 image_only 20260727 &
pid3="$!"
run_worker 4 prior_only 20260725 &
pid4="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  for mode in image_only prior_only; do
    "${PYTHON}" scripts/aggregate_sn7_changemamba.py \
      "${RUN_ROOT}/full_weight5_${mode}_three_seed_20260726.json" \
      "${RUN_ROOT}/full_weight5_${mode}_seed20260725_v1" \
      "${RUN_ROOT}/full_weight5_${mode}_seed20260726_v1" \
      "${RUN_ROOT}/full_weight5_${mode}_seed20260727_v1" \
      --output-markdown \
      "${RUN_ROOT}/full_weight5_${mode}_three_seed_20260726.md"
  done
  "${PYTHON}" scripts/compare_sn7_changemamba_modalities.py \
    "${RUN_ROOT}/full_weight5_three_seed_20260726.json" \
    "${RUN_ROOT}/full_weight5_image_only_three_seed_20260726.json" \
    "${RUN_ROOT}/full_weight5_prior_only_three_seed_20260726.json" \
    "${RUN_ROOT}/full_weight5_modality_comparison_20260726.json" \
    --output-markdown \
    "${RUN_ROOT}/full_weight5_modality_comparison_20260726.md"
fi
exit "${status}"
