#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

summaries=(
  "${RUN}/vla_features_sft_last_mean_train_n2048/summary.json"
  "${RUN}/vla_features_sft_last_mean_val_n1024/summary.json"
  "${RUN}/vla_features_sft_last_train_n2048/summary.json"
  "${RUN}/vla_features_sft_last_val_n1024/summary.json"
)
for summary in "${summaries[@]}"; do
  until [[ -s "${summary}" ]]; do
    sleep "${POLL_SECONDS}"
  done
done

"${PYTHON}" scripts/train_visual_utility_gate.py \
  "${RUN}/vla_features_sft_last_mean_train_n2048" \
  "${RUN}/vla_features_sft_last_mean_val_n1024" \
  "${RUN}/vla_utility_head_last_mean_pilot" \
  --seed 20260727 --max-call-rate 0.15 --min-oof-recall 0.10

"${PYTHON}" scripts/train_visual_utility_gate.py \
  "${RUN}/vla_features_sft_last_train_n2048" \
  "${RUN}/vla_features_sft_last_val_n1024" \
  "${RUN}/vla_utility_head_last_pilot" \
  --seed 20260727 --max-call-rate 0.15 --min-oof-recall 0.10
