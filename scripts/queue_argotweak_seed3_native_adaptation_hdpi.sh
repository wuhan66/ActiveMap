#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
OFFICIAL="${OFFICIAL:-${STORE}/external/ArgoTweak_baselines}"
PYTHON="${PYTHON:-${STORE}/envs/argotweak-legacy/bin/python}"
DATA="${STORE}/datasets/argotweak/tbv_balanced_24_8_v1/official_full"
SEED="${ARGOTWEAK_SEED:-20260833}"
GPU_IDS="${GPU_IDS:-1,3,4,5}"
MASTER_PORT="${MASTER_PORT:-29533}"
POLL_SECONDS="${POLL_SECONDS:-120}"
FREE_MEMORY_LIMIT_MIB="${FREE_MEMORY_LIMIT_MIB:-1000}"

RUN="${STORE}/runs/argotweak/full_domain_adaptation_balanced24_v1/four_gpu_seed${SEED}"
QUEUE_ROOT="${STORE}/runs/argotweak/seed3_native_adaptation_queue_v1"
EVAL_ROOT="${STORE}/runs/argotweak/seed3_epoch10_official_eval_v1"
TRAIN_LOG="${STORE}/logs/argotweak_full_domain_adaptation_balanced24_seed${SEED}.log"
QUEUE_LOG="${QUEUE_ROOT}/queue.log"

mkdir -p "${QUEUE_ROOT}" "$(dirname "${TRAIN_LOG}")"
exec 9>"${QUEUE_ROOT}/.queue.lock"
flock -n 9 || exit 0

IFS=',' read -r -a GPUS <<<"${GPU_IDS}"
if [[ "${#GPUS[@]}" -ne 4 ]]; then
  echo "GPU_IDS must contain exactly four physical GPU ids" >&2
  exit 2
fi
for gpu in "${GPUS[@]}"; do
  if [[ "${gpu}" == "0" || "${gpu}" == "2" ]]; then
    echo "GPU ${gpu} is reserved and cannot be used" >&2
    exit 2
  fi
done

for path in \
  "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
  "${OFFICIAL}/checkpoints/argotweak_baseline.pth" \
  "${DATA}/train_argotweak_balanced24.pkl" \
  "${DATA}/val_argotweak_balanced8.pkl"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done

gpu_set_is_free() {
  local gpu used
  for gpu in "${GPUS[@]}"; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu}" | tr -d ' ')"
    [[ "${used}" =~ ^[0-9]+$ ]] || return 1
    (( used <= FREE_MEMORY_LIMIT_MIB )) || return 1
  done
}

if [[ ! -e "${RUN}/TRAINING_COMPLETED" ]]; then
  if [[ -e "${RUN}" ]]; then
    echo "refusing partial or pre-existing run without TRAINING_COMPLETED: ${RUN}" >&2
    exit 4
  fi

  printf '%s waiting for GPUs %s\n' "$(date -Is)" "${GPU_IDS}" | tee -a "${QUEUE_LOG}"
  while ! gpu_set_is_free; do
    printf '%s GPUs still occupied; retrying in %ss\n' "$(date -Is)" "${POLL_SECONDS}" >>"${QUEUE_LOG}"
    sleep "${POLL_SECONDS}"
  done

  mkdir -p "${RUN}"
  "${PYTHON}" - "${RUN}/protocol.json" "${SEED}" "${GPU_IDS}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "schema_version": "activemap-argotweak-full-domain-adaptation-v2",
    "train_logs": 24,
    "val_logs": 8,
    "official_frames": 3615,
    "initial_checkpoint_sha256": "c42c29a88223db900464cfa88ef2a6f6c5575e5baf899b9c01a07bd674c31ce9",
    "epochs": 10,
    "physical_gpus": [int(value) for value in sys.argv[3].split(",")],
    "samples_per_gpu": 1,
    "auto_scale_lr_base_batch_size": 8,
    "seed": int(sys.argv[2]),
    "split": "train+validation",
    "test_assets_read": False,
}, indent=2) + "\n")
PY

  cd "${PROJECT_ROOT}"
  export PYTHONPATH="${OFFICIAL}:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
  export OMP_NUM_THREADS=2
  export MKL_NUM_THREADS=2

  printf '%s starting four-GPU seed %s\n' "$(date -Is)" "${SEED}" | tee -a "${QUEUE_LOG}"
  if "${PYTHON}" -m torch.distributed.launch \
    --nproc_per_node=4 --master_port="${MASTER_PORT}" \
    "${PROJECT_ROOT}/scripts/run_argotweak_official_train.py" \
    --official-root "${OFFICIAL}" \
    --config "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
    --work-dir "${RUN}" \
    --train-ann "${DATA}/train_argotweak_balanced24.pkl" \
    --val-ann "${DATA}/val_argotweak_balanced8.pkl" \
    --checkpoint "${OFFICIAL}/checkpoints/argotweak_baseline.pth" \
    --epochs 10 --workers 4 --seed "${SEED}" \
    --launcher pytorch --autoscale-lr >"${TRAIN_LOG}" 2>&1; then
    date -Is >"${RUN}/TRAINING_COMPLETED"
  else
    date -Is >"${RUN}/TRAINING_FAILED"
    exit 1
  fi
fi

if [[ ! -e "${EVAL_ROOT}/MATRIX_COMPLETED" ]]; then
  [[ -s "${RUN}/epoch_10.pth" ]] || { echo "missing trained checkpoint" >&2; exit 5; }
  [[ ! -e "${EVAL_ROOT}" ]] || { echo "refusing partial evaluation: ${EVAL_ROOT}" >&2; exit 6; }
  mkdir -p "${EVAL_ROOT}/evaluation"
  cd "${PROJECT_ROOT}"
  export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${OFFICIAL}${PYTHONPATH:+:${PYTHONPATH}}"
  CUDA_VISIBLE_DEVICES="${GPUS[0]}" "${PYTHON}" -u scripts/run_argotweak_official_test.py \
    --official-root "${OFFICIAL}" \
    --config "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
    --annotations "${DATA}/val_argotweak_balanced8.pkl" \
    --checkpoint "${RUN}/epoch_10.pth" \
    --output-dir "${EVAL_ROOT}/evaluation" --workers 4 --seed "${SEED}" \
    >"${EVAL_ROOT}/evaluation.log" 2>&1
  "${PYTHON}" scripts/export_argotweak_official_proposals.py \
    --results "${EVAL_ROOT}/evaluation/results.pkl" \
    --annotations "${DATA}/val_argotweak_balanced8.pkl" \
    --output "${EVAL_ROOT}/atomic_edit_proposals.jsonl" \
    --object-threshold 0.3 --object-match-distance 1.5 \
    >>"${EVAL_ROOT}/evaluation.log" 2>&1
  touch "${EVAL_ROOT}/MATRIX_COMPLETED"
fi

date -Is >"${QUEUE_ROOT}/QUEUE_COMPLETED"
printf '%s queue complete\n' "$(date -Is)" | tee -a "${QUEUE_LOG}"
