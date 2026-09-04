#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${STORAGE_ROOT}/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed3/seed20260719/final"
DATA="${STORAGE_ROOT}/runs/sn7_active_catalog/active_catalog_rl_states"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
LOG="${STORAGE_ROOT}/logs"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" scripts/extract_active_catalog_vla_features.py \
  "${MODEL}" "${ADAPTER}" "${DATA}/train.jsonl" \
  "${RUN}/vla_features_sft_last_mean_train_n2048" \
  --pooling last_mean --batch-size 2 --limit 2048 \
  > "${LOG}/vla_features_sft_last_mean_train_n2048.log" 2>&1 &
pid1="$!"

CUDA_VISIBLE_DEVICES=2 "${PYTHON}" scripts/extract_active_catalog_vla_features.py \
  "${MODEL}" "${ADAPTER}" "${DATA}/val.jsonl" \
  "${RUN}/vla_features_sft_last_mean_val_n1024" \
  --pooling last_mean --batch-size 2 --limit 1024 \
  > "${LOG}/vla_features_sft_last_mean_val_n1024.log" 2>&1 &
pid2="$!"

CUDA_VISIBLE_DEVICES=4 "${PYTHON}" scripts/extract_active_catalog_vla_features.py \
  "${MODEL}" "${ADAPTER}" "${DATA}/train.jsonl" \
  "${RUN}/vla_features_sft_last_train_n2048" \
  --pooling last --batch-size 2 --limit 2048 \
  > "${LOG}/vla_features_sft_last_train_n2048.log" 2>&1 &
pid3="$!"

CUDA_VISIBLE_DEVICES=6 "${PYTHON}" scripts/extract_active_catalog_vla_features.py \
  "${MODEL}" "${ADAPTER}" "${DATA}/val.jsonl" \
  "${RUN}/vla_features_sft_last_val_n1024" \
  --pooling last --batch-size 2 --limit 1024 \
  > "${LOG}/vla_features_sft_last_val_n1024.log" 2>&1 &
pid4="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
