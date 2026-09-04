#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_active_block4_buffer1_v2_512"
OPENCD="${STORE}/external/open_cd"
CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
PRETRAINED="${STORE}/models/opencd_ban/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_active_block4_buffer1_v2_512"
LOGS="${STORE}/logs/spacenet8_active_v2_ban_modality"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"
run_one() {
  local gpu="$1" mode="$2" seed="$3"
  local label="ban_mit_${mode}_weight5_seed${seed}"
  local output="${ROOT}/${label}"
  [[ ! -s "${output}/summary.json" ]] || return
  [[ ! -e "$output" ]] || { echo "Refusing incomplete output: $output" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$CONFIG" "$output" \
    --model-name "$label" --pretrained-checkpoint "$PRETRAINED" \
    --mode "$mode" --device cuda:0 --epochs 40 --min-epochs 12 --patience 8 \
    --batch-size 1 --workers 3 --learning-rate 0.00006 \
    --positive-weight 5 --seed "$seed" >"${LOGS}/${label}.log" 2>&1
}

pids=()
run_one 1 post_only 20260802 & pids+=("$!")
run_one 2 post_only 20260803 & pids+=("$!")
run_one 3 post_only 20260804 & pids+=("$!")
run_one 4 pre_only 20260802 & pids+=("$!")
run_one 5 pre_only 20260803 & pids+=("$!")
run_one 7 pre_only 20260804 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
