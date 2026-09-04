#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
SFT_ADAPTER="${STORAGE_ROOT}/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed3/seed20260719/final"
RL_DATA="${STORAGE_ROOT}/runs/sn7_active_catalog/active_catalog_rl_states"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

run_variant() {
  local gpu="$1"
  local name="$2"
  local acquire_fraction="$3"
  local pairwise_weight="$4"
  "${PYTHON}" scripts/launch_active_catalog_vlm_rl.py \
    "${SFT_ADAPTER}" "${RL_DATA}/train.jsonl" "${RL_DATA}/val.jsonl" \
    "${RUN_ROOT}/${name}" \
    --gpu "${gpu}" --seed 20260717 --epochs 1 \
    --learning-rate 5e-6 --gradient-accumulation 16 \
    --max-length 2048 --action-limit 4 --kl-beta 0.05 \
    --entropy-weight 0.01 --temperature 1.0 --utility-scale 30 \
    --acquire-fraction "${acquire_fraction}" \
    --pairwise-weight "${pairwise_weight}" --pairwise-margin 0.2 \
    --length-normalize-action-score \
    --logging-steps 10 --eval-steps 32 --save-steps 32 \
    --max-train-samples 512 --max-eval-samples 128 --monitor-interval 5
}

run_variant 1 contextual_rl_len_norm_n512 0.0 0.0 &
pid1="$!"
run_variant 2 contextual_rl_len_norm_bal50_n512 0.5 0.0 &
pid2="$!"
run_variant 4 contextual_rl_len_norm_pair01_n512 0.0 0.1 &
pid3="$!"
run_variant 6 contextual_rl_len_norm_bal50_pair01_n512 0.5 0.1 &
pid4="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
