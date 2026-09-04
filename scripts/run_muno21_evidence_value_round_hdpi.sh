#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/wh/projects/activemap-v1}"
STORAGE="${STORAGE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE}/envs/activemap-agent/bin/python}"
EPISODES="${EPISODES:-${STORAGE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
UPDATER="${UPDATER:-${STORAGE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
STATES_ROOT="${STATES_ROOT:-${STORAGE}/processed/muno21_v2/agent/executable_value_v1}"
STATES="${STATES:-${STATES_ROOT}/states_train_val_executable_balanced_512.jsonl}"
RUN_ROOT="${RUN_ROOT:-${STORAGE}/runs/muno21_evidence_value_v1_20260725}"
BUILD_GPU="${BUILD_GPU:-0}"
TRAIN_GPUS="${TRAIN_GPUS:-0,2,3}"
STAGE="${1:-all}"

require_file() {
  [[ -s "$1" ]] || { echo "Required input is missing: $1" >&2; exit 3; }
}

require_free_gpu() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    echo "GPU ${gpu} has active compute processes: ${pids}" >&2
    exit 4
  fi
}

build_states() {
  require_file "${EPISODES}"
  require_file "${UPDATER}"
  if [[ -s "${STATES}" && -s "${STATES%.jsonl}.summary.json" ]]; then
    echo "Executable states already complete: ${STATES}"
    return
  fi
  [[ ! -e "${STATES}" ]] || {
    echo "Refusing partial state file: ${STATES}" >&2
    exit 5
  }
  require_free_gpu "${BUILD_GPU}"
  mkdir -p "${STATES_ROOT}" "${RUN_ROOT}"
  cd "${REPO}"
  CUDA_VISIBLE_DEVICES="${BUILD_GPU}" PYTHONPATH=src:. "${PYTHON}" \
    -m activemap.cli build-selector-oracle \
    "${UPDATER}" "${EPISODES}" "${STATES}" \
    --device cuda:0 \
    --image-size 512 \
    --cost-weight 0.02 \
    --false-edit-weight 0.35 \
    --utility-mode executable \
    --utility-profile balanced \
    --writeback-threshold 0.5 \
    --writeback-delta-margin 0.0 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE}" \
    --budgets 1.5,3.0,4.5 \
    --splits train,val \
    2>&1 | tee "${RUN_ROOT}/build_states.log"
  "${PYTHON}" scripts/audit_selector_states.py "${STATES}" \
    --output "${STATES_ROOT}/states.audit.json"
  "${PYTHON}" scripts/audit_selector_utility_structure.py "${STATES}" \
    --output "${STATES_ROOT}/states.utility_audit.json"
}

launch_train() {
  require_file "${STATES}"
  IFS=',' read -r -a gpus <<<"${TRAIN_GPUS}"
  local seeds=(20260725 20260726 20260727)
  [[ "${#gpus[@]}" -ge "${#seeds[@]}" ]] || {
    echo "TRAIN_GPUS must provide at least three GPUs" >&2
    exit 6
  }
  mkdir -p "${RUN_ROOT}"
  local index
  for index in "${!seeds[@]}"; do
    local gpu="${gpus[$index]}"
    local seed="${seeds[$index]}"
    local output="${RUN_ROOT}/seed$((index + 1))"
    if [[ -s "${output}/run/best.pt" ]]; then
      echo "seed$((index + 1)) already complete"
      continue
    fi
    [[ ! -e "${output}/run" ]] || {
      echo "Refusing partial training directory: ${output}/run" >&2
      exit 7
    }
    require_free_gpu "${gpu}"
    mkdir -p "${output}"
    (
      cd "${REPO}"
      CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. nohup "${PYTHON}" \
        scripts/train_evidence_value_head.py \
        "${STATES}" "${output}/run" \
        --device cuda:0 \
        --seed "${seed}" \
        --epochs 40 \
        --batch-size 128 \
        >"${output}/train.log" 2>&1 < /dev/null &
      echo "$!" >"${output}/pid"
    )
    echo "seed$((index + 1)): pid=$(cat "${output}/pid") gpu=${gpu}"
  done
}

status() {
  if [[ -s "${STATES%.jsonl}.summary.json" ]]; then
    echo "READY states ${STATES}"
  elif [[ -s "${STATES}.progress.json" ]]; then
    cat "${STATES}.progress.json"
  elif [[ -s "${STATES%.jsonl}.progress.json" ]]; then
    cat "${STATES%.jsonl}.progress.json"
  else
    echo "MISSING states ${STATES}"
  fi
  local seed
  for seed in 1 2 3; do
    local output="${RUN_ROOT}/seed${seed}"
    if [[ -s "${output}/run/best.pt" ]]; then
      echo "READY seed${seed}"
    elif [[ -s "${output}/train.log" ]]; then
      tail -n 1 "${output}/train.log"
    else
      echo "MISSING seed${seed}"
    fi
  done
}

case "${STAGE}" in
  build) build_states ;;
  train) launch_train ;;
  all)
    build_states
    launch_train
    ;;
  status) status ;;
  *)
    echo "Usage: $0 {build|train|all|status}" >&2
    exit 2
    ;;
esac
