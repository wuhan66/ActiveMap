#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_smoke80_v1"
OPENCD="${STORE}/external/open_cd"
CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
PRETRAINED="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
ROOT="${STORE}/runs/spacenet8_germany/opencd_smoke80_v1"
LOGS="${STORE}/logs/spacenet8_opencd_smoke80_v1"

mkdir -p "${ROOT}" "${LOGS}"
cd "${PROJECT}"

run_one() {
  local gpu="$1" mode="$2"
  local output="${ROOT}/${mode}_seed20260802"
  [[ ! -e "${output}" ]] || { echo "Refusing existing output: ${output}" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. nohup "${PYTHON}" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "${OPENCD}" "${CONFIG}" "${output}" \
    --pretrained-checkpoint "${PRETRAINED}" --mode "${mode}" --device cuda:0 \
    --epochs 40 --min-epochs 12 --patience 8 --batch-size 2 --workers 2 \
    --learning-rate 0.00006 --positive-weight 20 --seed 20260802 \
    >"${LOGS}/${mode}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${mode}.pid"
  echo "${mode}: GPU ${gpu}, PID $!"
}

run_one 1 pre_post
run_one 3 post_only
run_one 4 pre_only
