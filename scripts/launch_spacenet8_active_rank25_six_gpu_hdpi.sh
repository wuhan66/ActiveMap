#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_active_rank25_block4_buffer1_512"
OPENCD="${STORE}/external/open_cd"
CHANGE_CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BAN_CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
MIT="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
CLIP="${STORE}/models/opencd_ban/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_active_rank25_512"
LOGS="${STORE}/logs/spacenet8_active_rank25"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"
run_one() {
  local gpu="$1" backend="$2" seed="$3"
  local config pretrained batch
  if [[ "$backend" == changeformer ]]; then
    config="$CHANGE_CONFIG"; pretrained="$MIT"; batch=3
  else
    config="$BAN_CONFIG"; pretrained="$CLIP"; batch=1
  fi
  local label="${backend}_weight5_seed${seed}"
  local output="${ROOT}/${label}"
  [[ ! -s "${output}/summary.json" ]] || return
  [[ ! -e "$output" ]] || return
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$config" "$output" \
    --model-name "$label" --pretrained-checkpoint "$pretrained" \
    --mode pre_post --device cuda:0 --epochs 40 --min-epochs 12 --patience 8 \
    --batch-size "$batch" --workers 3 --learning-rate 0.00006 \
    --positive-weight 5 --seed "$seed" >"${LOGS}/${label}.log" 2>&1
}

pids=()
run_one 1 changeformer 20260802 & pids+=("$!")
run_one 2 changeformer 20260803 & pids+=("$!")
run_one 3 changeformer 20260804 & pids+=("$!")
run_one 4 ban_mit 20260802 & pids+=("$!")
run_one 5 ban_mit 20260803 & pids+=("$!")
run_one 7 ban_mit 20260804 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done
