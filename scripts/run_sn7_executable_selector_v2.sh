#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-status}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${PHYSICAL_GPU:-0}"
ROOT="${EXECUTABLE_SELECTOR_ROOT:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v2}"
SOURCE_EPISODES="${SN7_EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full/episodes_train_val.jsonl}"
AVAILABLE_EPISODES="${ROOT}/episodes_train_val_available.jsonl"
FULL_AVAILABLE_EPISODES="${ROOT}/episodes_train_val_full_assets.jsonl"
UPDATER="${SN7_UPDATER_CHECKPOINT:-${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt}"
VAL_STATES="${ROOT}/states_val_executable_balanced_m15.jsonl"
TRAIN_STATES="${ROOT}/states_train_executable_balanced_m15.jsonl"
LOG="${ROOT}/logs/build_val_executable_balanced_m15.log"
RESTORE_LOG="${ROOT}/logs/restore_and_build_train.log"
ASSET_MAP="/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

require_file() {
  [[ -s "$1" ]] || { echo "Required input is missing: $1" >&2; exit 3; }
}

require_gpu() {
  local pids
  pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "Physical GPU ${GPU} has active compute processes: ${pids}" >&2
    exit 4
  }
  export CUDA_VISIBLE_DEVICES="${GPU}"
}

case "${STAGE}" in
  filter)
    require_file "${SOURCE_EPISODES}"
    mkdir -p "${ROOT}"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/filter_episodes_by_assets.py" \
      "${SOURCE_EPISODES}" "${AVAILABLE_EPISODES}" \
      --splits train,val --asset-root-map "${ASSET_MAP}"
    ;;
  oracle_val)
    require_file "${AVAILABLE_EPISODES}"
    require_file "${UPDATER}"
    [[ ! -e "${VAL_STATES}" ]] || {
      echo "Refusing to overwrite ${VAL_STATES}" >&2
      exit 5
    }
    require_gpu
    mkdir -p "$(dirname "${LOG}")"
    cd "${PROJECT_ROOT}"
    "${PYTHON}" -m activemap.cli build-selector-oracle \
      "${UPDATER}" "${AVAILABLE_EPISODES}" "${VAL_STATES}" \
      --device cuda:0 --image-size 128 --budgets 1.5,3.0,4.5 --splits val \
      --utility-mode executable --utility-profile balanced \
      --writeback-delta-margin 0.15 --asset-root-map "${ASSET_MAP}" \
      2>&1 | tee "${LOG}"
    "${PYTHON}" scripts/audit_selector_states.py "${VAL_STATES}" \
      --output "${VAL_STATES%.jsonl}.audit.json"
    "${PYTHON}" scripts/audit_selector_utility_structure.py "${VAL_STATES}" \
      --output "${VAL_STATES%.jsonl}.utility_audit.json"
    ;;
  restore_and_oracle_train)
    require_file "${SOURCE_EPISODES}"
    require_file "${UPDATER}"
    [[ ! -e "${FULL_AVAILABLE_EPISODES}" ]] || {
      echo "Refusing to overwrite ${FULL_AVAILABLE_EPISODES}" >&2
      exit 5
    }
    [[ ! -e "${TRAIN_STATES}" ]] || {
      echo "Refusing to overwrite ${TRAIN_STATES}" >&2
      exit 5
    }
    mkdir -p "$(dirname "${RESTORE_LOG}")"
    cd "${PROJECT_ROOT}"
    bash scripts/download_sn7.sh "${STORAGE_ROOT}/datasets/sn7" true
    "${PYTHON}" scripts/filter_episodes_by_assets.py \
      "${SOURCE_EPISODES}" "${FULL_AVAILABLE_EPISODES}" \
      --splits train,val --asset-root-map "${ASSET_MAP}"
    require_gpu
    "${PYTHON}" -m activemap.cli build-selector-oracle \
      "${UPDATER}" "${FULL_AVAILABLE_EPISODES}" "${TRAIN_STATES}" \
      --device cuda:0 --image-size 128 --budgets 1.5,3.0,4.5 --splits train \
      --utility-mode executable --utility-profile balanced \
      --writeback-delta-margin 0.15 --asset-root-map "${ASSET_MAP}"
    "${PYTHON}" scripts/audit_selector_states.py "${TRAIN_STATES}" \
      --output "${TRAIN_STATES%.jsonl}.audit.json"
    "${PYTHON}" scripts/audit_selector_utility_structure.py "${TRAIN_STATES}" \
      --output "${TRAIN_STATES%.jsonl}.utility_audit.json"
    ;;
  status)
    echo "root=${ROOT}"
    [[ -f "${VAL_STATES%.jsonl}.progress.json" ]] && cat "${VAL_STATES%.jsonl}.progress.json"
    [[ -f "${VAL_STATES%.jsonl}.summary.json" ]] && cat "${VAL_STATES%.jsonl}.summary.json"
    [[ -f "${VAL_STATES%.jsonl}.audit.json" ]] && cat "${VAL_STATES%.jsonl}.audit.json"
    [[ -f "${FULL_AVAILABLE_EPISODES%.jsonl}.summary.json" ]] && \
      cat "${FULL_AVAILABLE_EPISODES%.jsonl}.summary.json"
    [[ -f "${TRAIN_STATES%.jsonl}.progress.json" ]] && \
      cat "${TRAIN_STATES%.jsonl}.progress.json"
    [[ -f "${TRAIN_STATES%.jsonl}.audit.json" ]] && \
      cat "${TRAIN_STATES%.jsonl}.audit.json"
    [[ -f "${STORAGE_ROOT}/datasets/sn7/SN7_buildings_train.tar.gz" ]] && \
      stat -c 'archive_bytes=%s' "${STORAGE_ROOT}/datasets/sn7/SN7_buildings_train.tar.gz"
    nvidia-smi -i "${GPU}" --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader
    ;;
  *)
    echo "Usage: $0 {filter|oracle_val|restore_and_oracle_train|status}" >&2
    exit 2
    ;;
esac
