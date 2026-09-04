#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
ADAPTER="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
DATA="${RUN}/active_catalog_rl_states"
SEED=20260718

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

run_variant() {
  local gpu="$1"
  local name="$2"
  local learning_rate="$3"
  local kl_beta="$4"
  local acquire_fraction="$5"
  local output="${RUN}/${name}"
  [[ ! -e "${output}" ]] || {
    echo "refusing existing RL output: ${output}" >&2
    return 1
  }
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} is occupied: ${pids}" >&2
    return 1
  }
  "${PYTHON}" scripts/launch_active_catalog_vlm_rl.py \
    "${ADAPTER}" "${DATA}/train.jsonl" "${DATA}/val.jsonl" "${output}" \
    --gpu "${gpu}" --seed "${SEED}" --epochs 1 \
    --learning-rate "${learning_rate}" --gradient-accumulation 16 \
    --max-length 2048 --action-limit 4 --kl-beta "${kl_beta}" \
    --entropy-weight 0.01 --temperature 1.0 --utility-scale 30 \
    --acquire-fraction "${acquire_fraction}" \
    --pairwise-weight 0.1 --pairwise-margin 0.2 \
    --length-normalize-action-score \
    --logging-steps 10 --eval-steps 64 --save-steps 64 \
    --max-train-samples 4096 --max-eval-samples 512 --monitor-interval 5
}

run_variant 0 contextual_rl_conservative_lr2p5e6_kl005_bal50_n4096 \
  2.5e-6 0.05 0.5 &
pid0="$!"
run_variant 1 contextual_rl_conservative_lr2p5e6_kl010_bal50_n4096 \
  2.5e-6 0.10 0.5 &
pid1="$!"
run_variant 2 contextual_rl_conservative_lr5e6_kl010_bal50_n4096 \
  5e-6 0.10 0.5 &
pid2="$!"
run_variant 3 contextual_rl_conservative_lr5e6_kl005_bal25_n4096 \
  5e-6 0.05 0.25 &
pid3="$!"

status=0
for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
