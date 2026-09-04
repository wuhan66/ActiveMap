#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_block4_buffer1_v1_512"
OPENCD="${STORE}/external/open_cd"
CLIP="${STORE}/models/opencd_ban/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_ban_block4_buffer1_v1_512"
LOGS="${STORE}/logs/spacenet8_opencd_spatial_ban_block4_buffer1_v1_512"

cd "$PROJECT"
for index in 0 1 2; do
  gpu=$((index + 1))
  seed=$((20260802 + index))
  label="ban_mit_b0_local_v2_seed${seed}"
  output="${ROOT}/${label}"
  [[ ! -e "$output" ]] || { echo "Refusing existing output: $output" >&2; exit 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. nohup "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$CONFIG" "$output" \
    --model-name "$label" --pretrained-checkpoint "$CLIP" \
    --mode pre_post --device cuda:0 --epochs 40 --min-epochs 12 --patience 8 \
    --batch-size 1 --workers 3 --learning-rate 0.00006 \
    --positive-weight 10 --seed "$seed" \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "started=${label} gpu=${gpu} pid=$!"
done
