#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
LOG_ROOT="${STORAGE_ROOT}/logs/changemamba_modality_20260726"
POLL_SECONDS="${POLL_SECONDS:-120}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

until [[ -s "${RUN_ROOT}/full_weight5_modality_comparison_20260726.json" ]]; do
  echo "Waiting for completed three-seed modality comparison."
  sleep "${POLL_SECONDS}"
done

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
    sleep "${POLL_SECONDS}"
  done
}

audit_one() {
  local gpu="$1"
  local mode="$2"
  local seed="$3"
  local run="${RUN_ROOT}/full_weight5_${mode}_seed${seed}_v1"
  local output="${run}/val_audit"
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite audit: ${output}" >&2
    return 1
  fi
  wait_for_gpu "${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_sn7_changemamba.py \
    "${run}/best.pt" "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${output}" \
    --split val \
    --device cuda:0 \
    --batch-size 16 \
    --workers 8 \
    --input-mode "${mode}" \
    --no-save-masks \
    > "${LOG_ROOT}/${mode}_seed${seed}_audit.log" 2>&1
}

audit_worker() {
  local gpu="$1"
  shift
  while [[ "$#" -gt 0 ]]; do
    audit_one "${gpu}" "$1" "$2"
    shift 2
  done
}

audit_worker 1 image_only 20260725 prior_only 20260726 &
pid1="$!"
audit_worker 2 image_only 20260726 prior_only 20260727 &
pid2="$!"
audit_worker 3 image_only 20260727 &
pid3="$!"
audit_worker 4 prior_only 20260725 &
pid4="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

"${PYTHON}" scripts/bootstrap_sn7_changemamba_modalities.py \
  --image-prior \
    "${RUN_ROOT}/full_weight5_seed20260725_v1/val_audit" \
    "${RUN_ROOT}/full_weight5_seed20260726_v1/val_audit" \
    "${RUN_ROOT}/full_weight5_seed20260727_v1/val_audit" \
  --image-only \
    "${RUN_ROOT}/full_weight5_image_only_seed20260725_v1/val_audit" \
    "${RUN_ROOT}/full_weight5_image_only_seed20260726_v1/val_audit" \
    "${RUN_ROOT}/full_weight5_image_only_seed20260727_v1/val_audit" \
  --prior-only \
    "${RUN_ROOT}/full_weight5_prior_only_seed20260725_v1/val_audit" \
    "${RUN_ROOT}/full_weight5_prior_only_seed20260726_v1/val_audit" \
    "${RUN_ROOT}/full_weight5_prior_only_seed20260727_v1/val_audit" \
  --output \
    "${RUN_ROOT}/full_weight5_modality_aoi_bootstrap_20260726.json" \
  --output-markdown \
    "${RUN_ROOT}/full_weight5_modality_aoi_bootstrap_20260726.md" \
  --draws 5000 \
  --seed 20260726
