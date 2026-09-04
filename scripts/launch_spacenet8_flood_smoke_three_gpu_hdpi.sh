#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_smoke20_v1"
ROOT="${STORE}/runs/spacenet8_germany/flood_smoke20_v1"
LOGS="${STORE}/logs/spacenet8_flood_smoke20_v1"

mkdir -p "${ROOT}" "${LOGS}"
cd "${PROJECT}"

run_one() {
  local gpu="$1" mode="$2"
  local output="${ROOT}/${mode}_seed20260802"
  [[ ! -e "${output}" ]] || { echo "Refusing existing output: ${output}" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src nohup "${PYTHON}" \
    scripts/train_spacenet8_flood_smoke.py "${DATA}/manifest.jsonl" "${output}" \
    --mode "${mode}" --device cuda:0 --epochs 30 --patience 6 \
    --batch-size 2 --learning-rate 0.0003 --positive-weight 20 --seed 20260802 \
    >"${LOGS}/${mode}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${mode}.pid"
  echo "${mode}: GPU ${gpu}, PID $!"
}

run_one 1 pre_post
run_one 3 post_only
run_one 4 pre_only
