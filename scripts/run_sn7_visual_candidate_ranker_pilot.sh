#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_sft_v4"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
TRAIN_FEATURES="${RUN}/vla_features_sft_last_train_n2048"
VAL_FEATURES="${RUN}/vla_features_sft_last_val_n1024"
LOG="${STORAGE_ROOT}/logs"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

run_variant() {
  local gpu="$1"
  local name="$2"
  local hidden="$3"
  local dropout="$4"
  local pairwise="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/train_active_catalog_candidate_ranker.py \
    "${DATA}/train.jsonl" "${DATA}/train_evaluation_index.jsonl" \
    "${DATA}/val.jsonl" "${DATA}/val_evaluation_index.jsonl" \
    "${RUN}/${name}" \
    --device cuda --seed 20260727 --epochs 15 --patience 4 \
    --batch-size 128 --learning-rate 1e-4 \
    --hidden-dim "${hidden}" --dropout "${dropout}" \
    --regression-weight 1 --listwise-weight 1 \
    --pairwise-weight "${pairwise}" --gate-weight 0.25 \
    --positive-weight 4 --positive-sampling-fraction 0.25 \
    --maximum-false-call-rate 0.05 \
    --train-state-features "${TRAIN_FEATURES}" \
    --val-state-features "${VAL_FEATURES}" \
    > "${LOG}/${name}.log" 2>&1
}

run_variant 1 visual_candidate_ranker_h32_p0_pilot 32 0.5 0.0 &
pid1="$!"
run_variant 4 visual_candidate_ranker_h32_p05_pilot 32 0.5 0.5 &
pid2="$!"
run_variant 6 visual_candidate_ranker_h64_p1_pilot 64 0.3 1.0 &
pid3="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
