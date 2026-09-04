#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
STAGE="${1:-status}"
GPU="${PHYSICAL_GPU:-2}"

EPISODES="${SN7_EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/episodes_v3.jsonl}"
CHECKPOINT="${SN7_UPDATER_CHECKPOINT:-${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt}"
ROOT="${SN7_SEQUENTIAL_ROOT:-${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1}"
SMOKE_EPISODES="${ROOT}/smoke/episodes.jsonl"
SMOKE_STATES="${ROOT}/smoke/selector_states.jsonl"
PILOT_EPISODES="${ROOT}/pilot20/episodes.jsonl"
PILOT_STATES="${ROOT}/pilot20/selector_states.jsonl"
FULL_STATES="${ROOT}/selector_states_train_val.jsonl"
FULL_STATE_AUDIT="${ROOT}/selector_states_train_val.audit.json"
FULL_UTILITY_AUDIT="${ROOT}/selector_states_train_val.utility_audit.json"
FULL_MERGE_SUMMARY="${ROOT}/selector_states_train_val.merge_summary.json"
FULL_SFT="${ROOT}/full/active_catalog_sft_v4"
FULL_SFT_ARCHIVE="${ROOT}/full/active_catalog_sft_v4.tgz"
FULL_EPISODES="${ROOT}/full/episodes_train_val.jsonl"
SHARD_ROOT="${ROOT}/full_shards"
NUM_SHARDS="${SN7_ORACLE_SHARDS:-8}"
SEED="${SN7_SEQUENTIAL_SEED:-20260717}"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

require_file() {
  [[ -s "$1" ]] || { echo "Required input is missing: $1" >&2; exit 3; }
}

require_gpu() {
  [[ "${GPU}" =~ ^[0-9]+$ ]] || { echo "Invalid physical GPU: ${GPU}" >&2; exit 2; }
  local pids
  pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "Physical GPU ${GPU} has active compute processes: ${pids}" >&2
    exit 4
  }
  export CUDA_VISIBLE_DEVICES="${GPU}"
}

audit() {
  require_file "${EPISODES}"
  mkdir -p "${ROOT}"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/audit_episode_support.py" \
    "${EPISODES}" --splits train,val --output "${ROOT}/episodes.audit.json"
}

smoke_subset() {
  require_file "${EPISODES}"
  if [[ ! -s "${SMOKE_EPISODES}" ]]; then
    mkdir -p "$(dirname "${SMOKE_EPISODES}")"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/subsample_episodes.py" \
      "${EPISODES}" "${SMOKE_EPISODES}" --per-operation 2 --seed "${SEED}"
  fi
}

smoke_oracle() {
  smoke_subset
  require_file "${CHECKPOINT}"
  [[ ! -e "${SMOKE_STATES}" ]] || {
    echo "Refusing to overwrite smoke selector states: ${SMOKE_STATES}" >&2
    exit 5
  }
  require_gpu
  cd "${PROJECT_ROOT}"
  "${PYTHON}" -m activemap.cli build-selector-oracle \
    "${CHECKPOINT}" "${SMOKE_EPISODES}" "${SMOKE_STATES}" \
    --device cuda:0 --image-size 128 --budgets 1.5,3.0,4.5 --splits train,val
  "${PYTHON}" scripts/audit_selector_states.py "${SMOKE_STATES}" \
    --output "${SMOKE_STATES%.jsonl}.audit.json"
  "${PYTHON}" scripts/audit_selector_utility_structure.py "${SMOKE_STATES}" \
    --output "${SMOKE_STATES%.jsonl}.utility_audit.json"
}

pilot_subset() {
  require_file "${EPISODES}"
  if [[ ! -s "${PILOT_EPISODES}" ]]; then
    mkdir -p "$(dirname "${PILOT_EPISODES}")"
    "${PYTHON}" "${PROJECT_ROOT}/scripts/subsample_episodes.py" \
      "${EPISODES}" "${PILOT_EPISODES}" --per-operation 20 --seed "${SEED}"
  fi
}

pilot_oracle() {
  pilot_subset
  require_file "${CHECKPOINT}"
  [[ ! -e "${PILOT_STATES}" ]] || {
    echo "Refusing to overwrite pilot selector states: ${PILOT_STATES}" >&2
    exit 5
  }
  require_gpu
  cd "${PROJECT_ROOT}"
  "${PYTHON}" -m activemap.cli build-selector-oracle \
    "${CHECKPOINT}" "${PILOT_EPISODES}" "${PILOT_STATES}" \
    --device cuda:0 --image-size 128 --budgets 1.5,3.0,4.5 --splits train,val
  "${PYTHON}" scripts/audit_selector_states.py "${PILOT_STATES}" \
    --output "${PILOT_STATES%.jsonl}.audit.json"
  "${PYTHON}" scripts/audit_selector_utility_structure.py "${PILOT_STATES}" \
    --output "${PILOT_STATES%.jsonl}.utility_audit.json"
}

full_oracle() {
  audit
  require_file "${CHECKPOINT}"
  [[ ! -e "${FULL_STATES}" ]] || {
    echo "Refusing to overwrite full selector states: ${FULL_STATES}" >&2
    exit 5
  }
  require_gpu
  cd "${PROJECT_ROOT}"
  "${PYTHON}" -m activemap.cli build-selector-oracle \
    "${CHECKPOINT}" "${EPISODES}" "${FULL_STATES}" \
    --device cuda:0 --image-size 128 --budgets 1.5,3.0,4.5 --splits train,val
  "${PYTHON}" scripts/audit_selector_states.py "${FULL_STATES}" \
    --output "${FULL_STATES%.jsonl}.audit.json"
  "${PYTHON}" scripts/audit_selector_utility_structure.py "${FULL_STATES}" \
    --output "${FULL_STATES%.jsonl}.utility_audit.json"
}

full_sharded_oracle() {
  audit
  require_file "${CHECKPOINT}"
  if [[ ! -s "${SHARD_ROOT}/summary.json" ]]; then
    "${PYTHON}" "${PROJECT_ROOT}/scripts/shard_episodes.py" \
      "${EPISODES}" "${SHARD_ROOT}" --num-shards "${NUM_SHARDS}" --seed "${SEED}"
  fi
  require_gpu
  cd "${PROJECT_ROOT}"
  local index shard episodes states
  for ((index=0; index<NUM_SHARDS; index++)); do
    shard="${SHARD_ROOT}/$(printf 'shard-%02d' "${index}")"
    episodes="${shard}/episodes.jsonl"
    states="${shard}/selector_states.jsonl"
    require_file "${episodes}"
    if [[ ! -s "${states}" ]]; then
      "${PYTHON}" -m activemap.cli build-selector-oracle \
        "${CHECKPOINT}" "${episodes}" "${states}" \
        --device cuda:0 --image-size 128 --budgets 1.5,3.0,4.5 --splits train,val
    fi
    if [[ ! -s "${states%.jsonl}.audit.json" ]]; then
      "${PYTHON}" scripts/audit_selector_states.py "${states}" \
        --output "${states%.jsonl}.audit.json"
    fi
  done
  if [[ ! -s "${FULL_STATES}" ]]; then
    "${PYTHON}" scripts/merge_selector_state_shards.py "${SHARD_ROOT}" "${FULL_STATES}"
  fi
  "${PYTHON}" scripts/audit_selector_states.py "${FULL_STATES}" \
    --output "${FULL_STATES%.jsonl}.audit.json"
  "${PYTHON}" scripts/audit_selector_utility_structure.py "${FULL_STATES}" \
    --output "${FULL_STATES%.jsonl}.utility_audit.json"
}

full_sft() {
  require_file "${FULL_STATES}"
  require_file "${FULL_STATE_AUDIT}"
  require_file "${FULL_UTILITY_AUDIT}"
  require_file "${FULL_MERGE_SUMMARY}"
  cd "${PROJECT_ROOT}"
  if [[ ! -s "${FULL_EPISODES}" ]]; then
    "${PYTHON}" scripts/merge_episode_shards.py "${SHARD_ROOT}" "${FULL_EPISODES}"
  fi
  require_file "${FULL_EPISODES}"
  if [[ ! -s "${FULL_SFT}/summary.json" ]]; then
    [[ ! -e "${FULL_SFT}" ]] || {
      echo "Refusing incomplete existing SFT directory: ${FULL_SFT}" >&2
      exit 6
    }
    "${PYTHON}" scripts/build_episode_sequential_selector_sft.py \
      "${FULL_STATES}" "${FULL_EPISODES}" "${FULL_SFT}" \
      --policy-snapshot sn7-updater-v4-vector-change-seed20260716 \
      --max-candidates 16
  fi
  if [[ ! -s "${FULL_SFT}/audit.json" ]]; then
    "${PYTHON}" scripts/audit_active_catalog_sft.py \
      "${FULL_SFT}" --output "${FULL_SFT}/audit.json"
  fi
  if [[ ! -s "${FULL_SFT}/smoke_train.jsonl" ]]; then
    "${PYTHON}" scripts/build_vlm_sft_smoke.py \
      "${FULL_SFT}/train.jsonl" "${FULL_SFT}/val.jsonl" \
      --train-output "${FULL_SFT}/smoke_train.jsonl" \
      --val-output "${FULL_SFT}/smoke_val.jsonl" \
      --summary "${FULL_SFT}/smoke_summary.json"
  fi
  if [[ ! -s "${FULL_SFT_ARCHIVE}" ]]; then
    tar -C "$(dirname "${FULL_SFT}")" -czf "${FULL_SFT_ARCHIVE}" \
      "$(basename "${FULL_SFT}")"
    sha256sum "${FULL_SFT_ARCHIVE}" > "${FULL_SFT_ARCHIVE}.sha256"
  fi
  cat "${FULL_SFT_ARCHIVE}.sha256"
}

status() {
  for path in "${EPISODES}" "${CHECKPOINT}" "${ROOT}/episodes.audit.json" \
    "${SMOKE_EPISODES}" "${SMOKE_STATES}" "${PILOT_EPISODES}" \
    "${PILOT_STATES}" "${SHARD_ROOT}/summary.json" "${FULL_STATES}"; do
    if [[ -s "${path}" ]]; then printf 'READY\t%s\n' "${path}"; else printf 'MISSING\t%s\n' "${path}"; fi
  done
  for path in "${FULL_MERGE_SUMMARY}" "${FULL_STATE_AUDIT}" \
    "${FULL_UTILITY_AUDIT}" "${FULL_EPISODES}" \
    "${FULL_EPISODES%.jsonl}.merge_summary.json" "${FULL_SFT}/summary.json" \
    "${FULL_SFT}/audit.json" "${FULL_SFT_ARCHIVE}"; do
    if [[ -s "${path}" ]]; then printf 'READY\t%s\n' "${path}"; else printf 'MISSING\t%s\n' "${path}"; fi
  done
}

case "${STAGE}" in
  audit) audit ;;
  smoke_subset) smoke_subset ;;
  smoke_oracle) smoke_oracle ;;
  pilot_subset) pilot_subset ;;
  pilot_oracle) pilot_oracle ;;
  full_oracle) full_oracle ;;
  full_sharded_oracle) full_sharded_oracle ;;
  full_sft) full_sft ;;
  status) status ;;
  *) echo "Usage: $0 {status|audit|smoke_subset|smoke_oracle|pilot_subset|pilot_oracle|full_oracle|full_sharded_oracle|full_sft}" >&2; exit 2 ;;
esac
