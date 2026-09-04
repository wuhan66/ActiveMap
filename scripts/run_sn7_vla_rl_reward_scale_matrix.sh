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
  local learning_rate="$3"
  local utility_scale="$4"
  "${PYTHON}" scripts/launch_active_catalog_vlm_rl.py \
    "${SFT_ADAPTER}" "${RL_DATA}/train.jsonl" "${RL_DATA}/val.jsonl" \
    "${RUN_ROOT}/${name}" \
    --gpu "${gpu}" --seed 20260717 --epochs 1 \
    --learning-rate "${learning_rate}" --gradient-accumulation 16 \
    --max-length 2048 --action-limit 4 --kl-beta 0.05 \
    --entropy-weight 0.01 --temperature 1.0 \
    --utility-scale "${utility_scale}" \
    --logging-steps 10 --eval-steps 32 --save-steps 32 \
    --max-train-samples 512 --max-eval-samples 128 --monitor-interval 5
}

run_variant 0 contextual_rl_diag_scale10_n512 5e-6 10 &
pid1="$!"
run_variant 2 contextual_rl_diag_scale30_n512 5e-6 30 &
pid2="$!"
run_variant 4 contextual_rl_diag_scale100_n512 5e-6 100 &
pid3="$!"
run_variant 6 contextual_rl_diag_scale30_lr1e5_n512 1e-5 30 &
pid4="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
