#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
SFT_ADAPTER="${RL_BASE_ADAPTER:-${STORAGE_ROOT}/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed3/seed20260719/final}"
RL_DATA="${STORAGE_ROOT}/runs/sn7_active_catalog/active_catalog_rl_states"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog"
TRAIN_SAMPLES="${RL_DIAGNOSTIC_TRAIN_SAMPLES:-128}"
EVAL_SAMPLES="${RL_DIAGNOSTIC_EVAL_SAMPLES:-64}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

run_variant() {
  local gpu="$1"
  local name="$2"
  local learning_rate="$3"
  local kl_beta="$4"
  local entropy_weight="$5"
  "${PYTHON}" scripts/launch_active_catalog_vlm_rl.py \
    "${SFT_ADAPTER}" "${RL_DATA}/train.jsonl" "${RL_DATA}/val.jsonl" \
    "${RUN_ROOT}/${name}" \
    --gpu "${gpu}" --seed 20260717 --epochs 1 \
    --learning-rate "${learning_rate}" --gradient-accumulation 16 \
    --max-length 2048 --action-limit 4 --kl-beta "${kl_beta}" \
    --entropy-weight "${entropy_weight}" --temperature 1.0 \
    --logging-steps 5 --eval-steps 64 --save-steps 64 \
    --max-train-samples "${TRAIN_SAMPLES}" \
    --max-eval-samples "${EVAL_SAMPLES}" --monitor-interval 5
}

run_variant 2 contextual_rl_diag_lr1e6_kl005_ent001_n128 1e-6 0.05 0.01 &
pid1="$!"
run_variant 3 contextual_rl_diag_lr5e6_kl005_ent001_n128 5e-6 0.05 0.01 &
pid2="$!"
run_variant 4 contextual_rl_diag_lr1e6_kl010_ent001_n128 1e-6 0.10 0.01 &
pid3="$!"
run_variant 6 contextual_rl_diag_lr1e6_kl005_ent000_n128 1e-6 0.05 0.00 &
pid4="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}" "${pid4}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
