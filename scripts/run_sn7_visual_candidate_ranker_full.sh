#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${STORAGE_ROOT}/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed3/seed20260719/final"
RL="${STORAGE_ROOT}/runs/sn7_active_catalog/active_catalog_rl_states"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_sft_v4"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
LOG="${STORAGE_ROOT}/logs"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

extract_gpu() {
  local gpu="$1"
  local shard="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/extract_active_catalog_vla_features.py \
    "${MODEL}" "${ADAPTER}" "${RL}/train.jsonl" \
    "${RUN}/vla_features_last_train_full_s${shard}of3" \
    --pooling last --batch-size 2 --num-shards 3 --shard-index "${shard}" \
    > "${LOG}/vla_features_last_train_full_s${shard}of3.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/extract_active_catalog_vla_features.py \
    "${MODEL}" "${ADAPTER}" "${RL}/val.jsonl" \
    "${RUN}/vla_features_last_val_full_s${shard}of3" \
    --pooling last --batch-size 2 --num-shards 3 --shard-index "${shard}" \
    > "${LOG}/vla_features_last_val_full_s${shard}of3.log" 2>&1
}

extract_gpu 1 0 & pid1="$!"
extract_gpu 4 1 & pid2="$!"
extract_gpu 6 2 & pid3="$!"
status=0
for pid in "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then status=1; fi
done
if [[ "${status}" -ne 0 ]]; then exit "${status}"; fi

"${PYTHON}" scripts/merge_active_catalog_vla_features.py \
  "${RUN}/vla_features_last_train_full" \
  "${RUN}/vla_features_last_train_full_s0of3" \
  "${RUN}/vla_features_last_train_full_s1of3" \
  "${RUN}/vla_features_last_train_full_s2of3"
"${PYTHON}" scripts/merge_active_catalog_vla_features.py \
  "${RUN}/vla_features_last_val_full" \
  "${RUN}/vla_features_last_val_full_s0of3" \
  "${RUN}/vla_features_last_val_full_s1of3" \
  "${RUN}/vla_features_last_val_full_s2of3"

train_variant() {
  local gpu="$1"
  local name="$2"
  local hidden="$3"
  local dropout="$4"
  local pairwise="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_active_catalog_candidate_ranker.py \
    "${DATA}/train.jsonl" "${DATA}/train_evaluation_index.jsonl" \
    "${DATA}/val.jsonl" "${DATA}/val_evaluation_index.jsonl" \
    "${RUN}/${name}" \
    --device cuda --seed 20260727 --epochs 20 --patience 5 \
    --batch-size 256 --learning-rate 1e-4 \
    --hidden-dim "${hidden}" --dropout "${dropout}" \
    --regression-weight 1 --listwise-weight 1 \
    --pairwise-weight "${pairwise}" --gate-weight 0.25 \
    --positive-weight 4 --positive-sampling-fraction 0.25 \
    --maximum-false-call-rate 0.05 \
    --train-state-features "${RUN}/vla_features_last_train_full" \
    --val-state-features "${RUN}/vla_features_last_val_full" \
    > "${LOG}/${name}.log" 2>&1
}

train_variant 1 visual_candidate_ranker_full_h32_p0 32 0.5 0.0 & pid1="$!"
train_variant 4 visual_candidate_ranker_full_h32_p05 32 0.5 0.5 & pid2="$!"
train_variant 6 visual_candidate_ranker_full_h64_p1 64 0.3 1.0 & pid3="$!"
status=0
for pid in "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then status=1; fi
done
exit "${status}"
