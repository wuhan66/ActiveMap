#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
STORE="${ACTIVEMAP_STORE:-/home/wh/ActiveMap}"
SHARD_ROOT="${SHARD_ROOT:-${STORE}/processed/sn7_v1/agent/executable_selector_v3_512_sharded}"
CHECKPOINT="${CHECKPOINT:-${STORE}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORE}/runs/selector/sn7_mask_features_v2_full_20260901}"
GPUS_CSV="${GPUS_CSV:-2,3,4,5}"
IFS=',' read -r -a GPUS <<< "${GPUS_CSV}"

if (( ${#GPUS[@]} == 0 )); then
  echo "GPUS_CSV must name at least one GPU" >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}/logs"

run_shard() {
  local gpu="$1"
  local shard_index="$2"
  local shard
  local episodes
  local output
  shard="$(printf 'shard-%02d' "${shard_index}")"
  episodes="${SHARD_ROOT}/${shard}/episodes.jsonl"
  output="${OUTPUT_ROOT}/${shard}_states.jsonl"
  if [[ -s "${output}" && -s "${output%.jsonl}.summary.json" ]]; then
    echo "[skip] ${shard} already complete"
    return
  fi
  if [[ -e "${output}" || -e "${output}.partial" ]]; then
    echo "refusing ambiguous partial output for ${shard}" >&2
    exit 3
  fi
  echo "[start] ${shard} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m activemap build-selector-oracle \
    "${CHECKPOINT}" \
    "${episodes}" \
    "${output}" \
    --device cuda \
    --image-size 512 \
    --utility-mode executable \
    --utility-profile balanced \
    --writeback-threshold 0.5 \
    --writeback-delta-margin 0.15 \
    --asset-root-map /mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap \
    --budgets 3.0 \
    --max-steps 1 \
    --splits train,val \
    --candidate-workers 8
  echo "[done] ${shard}"
}

worker() {
  local worker_index="$1"
  local gpu="${GPUS[${worker_index}]}"
  local shard_index="${worker_index}"
  while (( shard_index < 5 )); do
    run_shard "${gpu}" "${shard_index}"
    shard_index=$((shard_index + ${#GPUS[@]}))
  done
}

pids=()
for worker_index in "${!GPUS[@]}"; do
  worker "${worker_index}" >"${OUTPUT_ROOT}/logs/worker-${worker_index}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

combined="${OUTPUT_ROOT}/states_train_val_budget3_mask_v2.jsonl"
if [[ ! -s "${combined}" ]]; then
  "${PYTHON}" scripts/merge_selector_state_files.py \
    "${combined}" \
    "${OUTPUT_ROOT}/shard-00_states.jsonl" \
    "${OUTPUT_ROOT}/shard-01_states.jsonl" \
    "${OUTPUT_ROOT}/shard-02_states.jsonl" \
    "${OUTPUT_ROOT}/shard-03_states.jsonl" \
    "${OUTPUT_ROOT}/shard-04_states.jsonl"
fi

runtime_grid="${OUTPUT_ROOT}/states_train_val_budget3_runtime_grid_mask_v2.jsonl"
if [[ ! -s "${runtime_grid}" ]]; then
  "${PYTHON}" scripts/build_runtime_grid_selector_manifest.py \
    "${combined}" \
    "${runtime_grid}" \
    --episodes "${SHARD_ROOT}/shard-00/episodes.jsonl" \
    --episodes "${SHARD_ROOT}/shard-01/episodes.jsonl" \
    --episodes "${SHARD_ROOT}/shard-02/episodes.jsonl" \
    --episodes "${SHARD_ROOT}/shard-03/episodes.jsonl" \
    --episodes "${SHARD_ROOT}/shard-04/episodes.jsonl" \
    --image-size 512 \
    --budget 3.0 \
    --asset-root-map /mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap
fi

for feature_set in policy-relative policy-relative-mask; do
  audit="${OUTPUT_ROOT}/learnability_${feature_set}.json"
  if [[ ! -s "${audit}" ]]; then
    "${PYTHON}" scripts/audit_selector_candidate_learnability.py \
      "${runtime_grid}" \
      "${audit}" \
      --feature-set "${feature_set}" \
      --model-type classifier \
      --positive-weight 32 \
      --max-iter 200 \
      --max-false-call-rate 0.02 \
      --max-harmful-call-fraction 0.20 \
      --min-acquire-recall 0.10
  fi
done

echo "SN7 mask_features-v2 full audit complete: ${OUTPUT_ROOT}"
