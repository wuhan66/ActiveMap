#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
MIGRATION_SUMMARY="${MIGRATION_SUMMARY:-${STORAGE_ROOT}/logs/sn7_migration_20260725.summary}"
DATA_ROOT="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20"
SOURCE_MANIFEST="${DATA_ROOT}/updater_samples.jsonl"
FROZEN_MANIFEST="${DATA_ROOT}/train_val_complete_20260726.jsonl"
AUDIT_REPORT="${STORAGE_ROOT}/logs/sn7_train_val_complete_20260726.audit.json"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CHANGE_CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
ENCODER="${STORAGE_ROOT}/models/change_mamba/vssm_tiny_0230_ckpt_epoch_262.pth"
ENCODER_URL="${ENCODER_URL:-https://zenodo.org/records/15479555/files/vssm_tiny_0230_ckpt_epoch_262.pth?download=1}"
ENCODER_MD5="d64653bba8f6e5c0d6f4ac6275e1be61"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
LOG_ROOT="${STORAGE_ROOT}/logs/changemamba_full_20260726"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-120}"

mkdir -p "${LOG_ROOT}" "$(dirname "${ENCODER}")"
cd "${PROJECT_ROOT}"

until [[ -f "${MIGRATION_SUMMARY}" ]] &&
  grep -qx "status=complete" "${MIGRATION_SUMMARY}"; do
  echo "Waiting for verified SN7 migration: ${MIGRATION_SUMMARY}"
  sleep 60
done

"${PYTHON}" scripts/audit_updater_asset_availability.py \
  "${SOURCE_MANIFEST}" \
  --splits train,val \
  --output "${FROZEN_MANIFEST}" \
  > "${AUDIT_REPORT}"

"${PYTHON}" -c '
import json
import sys
from pathlib import Path

report = json.loads(Path(sys.argv[1]).read_text())
available = int(report["available_count"])
expected = sum(int(value) for value in report["total_by_split_edit"].values())
if available != expected:
    raise SystemExit(f"incomplete train/val assets: {available}/{expected}")
if not report["available_by_split_edit"].get("val:RESHAPE", 0):
    raise SystemExit("validation split has no RESHAPE support")
print(f"Frozen train/val manifest: {available}/{expected} complete records")
' "${AUDIT_REPORT}"

if ! echo "${ENCODER_MD5}  ${ENCODER}" | md5sum --check --status; then
  curl --location --fail --retry 8 --retry-delay 10 \
    --continue-at - --output "${ENCODER}" "${ENCODER_URL}"
fi
echo "${ENCODER_MD5}  ${ENCODER}" | md5sum --check

gpu_is_idle() {
  local gpu="$1"
  local memory
  local utilization
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

declare -a pids=()
declare -a seeds=(20260725 20260726 20260727)
declare -a gpus=(0 2 3)

for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  gpu="${gpus[$index]}"
  output="${RUN_ROOT}/full_weight5_seed${seed}_v1"
  log="${LOG_ROOT}/seed${seed}.log"
  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing run: ${output}" >&2
    exit 1
  fi
  wait_for_gpu "${gpu}"
  echo "Launching seed ${seed} on physical GPU ${gpu}."
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_sn7_changemamba.py \
    "${FROZEN_MANIFEST}" \
    "${CHANGE_REPO}" \
    "${CHANGE_CONFIG}" \
    "${output}" \
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
    --seed "${seed}" \
    > "${log}" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -eq 0 ]]; then
  "${PYTHON}" scripts/aggregate_sn7_changemamba.py \
    "${RUN_ROOT}/full_weight5_three_seed_20260726.json" \
    "${RUN_ROOT}/full_weight5_seed20260725_v1" \
    "${RUN_ROOT}/full_weight5_seed20260726_v1" \
    "${RUN_ROOT}/full_weight5_seed20260727_v1" \
    --output-markdown \
    "${RUN_ROOT}/full_weight5_three_seed_20260726.md"
fi
exit "${status}"
