#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_block4_buffer1_v1_512"
OPENCD="${STORE}/external/open_cd"
CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
PRETRAINED="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_block4_buffer1_v1_512"
LOGS="${STORE}/logs/spacenet8_opencd_spatial_modality_block4_buffer1_v1_512"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_one() {
  local gpu="$1" mode="$2" seed="$3"
  local label="changeformer_${mode}_weight10_seed${seed}"
  local output="${ROOT}/${label}"
  if [[ -s "${output}/summary.json" ]]; then
    echo "skip_completed=${label}"
    return
  fi
  [[ ! -e "$output" ]] || { echo "Refusing incomplete existing output: $output" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. nohup "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$CONFIG" "$output" \
    --model-name ChangeFormer-MiT-B0-512 --pretrained-checkpoint "$PRETRAINED" \
    --mode "$mode" --device cuda:0 --epochs 40 --min-epochs 12 --patience 8 \
    --batch-size 4 --workers 4 --learning-rate 0.00006 \
    --positive-weight 10 --seed "$seed" \
    >"${LOGS}/${label}.log" 2>&1 < /dev/null &
  echo "$!" >"${LOGS}/${label}.pid"
  echo "started=${label} gpu=${gpu} pid=$!"
}

run_one 1 post_only 20260802
run_one 2 post_only 20260803
run_one 3 post_only 20260804
run_one 4 pre_only 20260802
run_one 5 pre_only 20260803
run_one 7 pre_only 20260804
