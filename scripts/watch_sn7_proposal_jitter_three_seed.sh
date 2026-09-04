#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/train_val_complete_20260726.jsonl"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
ENCODER="${STORAGE_ROOT}/models/change_mamba/vssm_tiny_0230_ckpt_epoch_262.pth"
RUN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
LOG_ROOT="${STORAGE_ROOT}/logs/changemamba_proposal_jitter_three_seed_20260726"
PILOT="${RUN_ROOT}/proposal_jitter16_pilot_20260726.json"
POLL_SECONDS="${POLL_SECONDS:-120}"
GPUS=(
  "${JITTER_GPU_IMAGE_PRIOR_SEED26:-2}"
  "${JITTER_GPU_IMAGE_PRIOR_SEED27:-3}"
  "${JITTER_GPU_PRIOR_ONLY_SEED26:-4}"
  "${JITTER_GPU_PRIOR_ONLY_SEED27:-6}"
)

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

until [[ -s "${PILOT}" ]]; do
  echo "Waiting for proposal-jitter pilot."
  sleep "${POLL_SECONDS}"
done

if ! "${PYTHON}" -c \
  "import json; assert json.load(open('${PILOT}'))['three_seed_promotion']"; then
  echo "Proposal-jitter pilot did not pass the predeclared promotion gate."
  exit 0
fi

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

run_and_audit() {
  local gpu="$1"
  local mode="$2"
  local seed="$3"
  local run="${RUN_ROOT}/proposal_jitter16_${mode}_seed${seed}_v1"
  local audit="${run}/val_audit"

  if [[ -e "${run}" ]]; then
    echo "Waiting for existing proposal-jitter run: ${run}"
    until [[ -s "${run}/summary.json" && -s "${run}/best.pt" ]]; do
      if ! pgrep -f "train_sn7_changemamba.py.*${run}" >/dev/null; then
        echo "Existing proposal-jitter run is incomplete and inactive: ${run}" >&2
        return 1
      fi
      sleep "${POLL_SECONDS}"
    done
  else
    wait_for_gpu "${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_sn7_changemamba.py \
      "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${run}" \
      --encoder-checkpoint "${ENCODER}" \
      --device cuda:0 --image-size 128 --batch-size 16 --workers 8 \
      --epochs 40 --min-epochs 10 --patience 8 \
      --learning-rate 0.0001 --positive-class-weight 5 \
      --max-translation-pixels 16 --input-mode "${mode}" --seed "${seed}" \
      > "${LOG_ROOT}/${mode}_seed${seed}.log" 2>&1
  fi

  if [[ ! -s "${audit}/summary.json" ]]; then
    if [[ -e "${audit}" ]]; then
      echo "Refusing partial proposal-jitter audit: ${audit}" >&2
      return 1
    fi
    wait_for_gpu "${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_sn7_changemamba.py \
      "${run}/best.pt" "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${audit}" \
      --split val --device cuda:0 --batch-size 16 --workers 8 \
      --input-mode "${mode}" --no-save-masks \
      > "${LOG_ROOT}/${mode}_seed${seed}_audit.log" 2>&1
  fi
}

run_and_audit "${GPUS[0]}" image_prior 20260726 &
pid1="$!"
run_and_audit "${GPUS[1]}" image_prior 20260727 &
pid2="$!"
run_and_audit "${GPUS[2]}" prior_only 20260726 &
pid3="$!"
run_and_audit "${GPUS[3]}" prior_only 20260727 &
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

audit_existing() {
  local gpu="$1"
  local mode="$2"
  local seed="$3"
  local run="${RUN_ROOT}/proposal_jitter16_${mode}_seed${seed}_v1"
  local audit="${run}/val_audit"
  if [[ -s "${audit}/summary.json" ]]; then
    return
  fi
  if [[ -e "${audit}" ]]; then
    echo "Refusing partial proposal-jitter audit: ${audit}" >&2
    return 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_sn7_changemamba.py \
    "${run}/best.pt" "${MANIFEST}" "${CHANGE_REPO}" "${CONFIG}" "${audit}" \
    --split val --device cuda:0 --batch-size 16 --workers 8 \
    --input-mode "${mode}" --no-save-masks \
    > "${LOG_ROOT}/${mode}_seed${seed}_audit.log" 2>&1
}

audit_existing "${GPUS[0]}" image_prior 20260725 &
audit1="$!"
audit_existing "${GPUS[1]}" prior_only 20260725 &
audit2="$!"
wait "${audit1}"
wait "${audit2}"

for mode in image_prior prior_only; do
  "${PYTHON}" scripts/aggregate_sn7_changemamba.py \
    "${RUN_ROOT}/proposal_jitter16_${mode}_three_seed_20260726.json" \
    "${RUN_ROOT}/proposal_jitter16_${mode}_seed20260725_v1" \
    "${RUN_ROOT}/proposal_jitter16_${mode}_seed20260726_v1" \
    "${RUN_ROOT}/proposal_jitter16_${mode}_seed20260727_v1" \
    --output-markdown \
      "${RUN_ROOT}/proposal_jitter16_${mode}_three_seed_20260726.md"
done

"${PYTHON}" scripts/bootstrap_sn7_updater_comparison.py \
  --baseline \
    "${RUN_ROOT}/proposal_jitter16_prior_only_seed20260725_v1/val_audit" \
    "${RUN_ROOT}/proposal_jitter16_prior_only_seed20260726_v1/val_audit" \
    "${RUN_ROOT}/proposal_jitter16_prior_only_seed20260727_v1/val_audit" \
  --candidate \
    "${RUN_ROOT}/proposal_jitter16_image_prior_seed20260725_v1/val_audit" \
    "${RUN_ROOT}/proposal_jitter16_image_prior_seed20260726_v1/val_audit" \
    "${RUN_ROOT}/proposal_jitter16_image_prior_seed20260727_v1/val_audit" \
  --baseline-name prior_only \
  --candidate-name image_prior \
  --output \
    "${RUN_ROOT}/proposal_jitter16_image_prior_vs_prior_only_aoi_bootstrap_20260726.json" \
  --output-markdown \
    "${RUN_ROOT}/proposal_jitter16_image_prior_vs_prior_only_aoi_bootstrap_20260726.md" \
  --draws 5000 --seed 20260726
