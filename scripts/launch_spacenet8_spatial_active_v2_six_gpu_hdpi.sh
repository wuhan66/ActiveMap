#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_active_block4_buffer1_v2_512"
OPENCD="${STORE}/external/open_cd"
CHANGE_CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BAN_CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
MIT="${STORE}/models/opencd_ban/mit_b0_20220624-7e0fe6dd.pth"
CLIP="${STORE}/models/opencd_ban/clip_vit-base-patch16-224_3rdparty-d08f8887.pth"
ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_active_block4_buffer1_v2_512"
LOGS="${STORE}/logs/spacenet8_opencd_spatial_active_block4_buffer1_v2_512"

mkdir -p "$ROOT" "$LOGS"
cd "$PROJECT"

run_one() {
  local gpu="$1" label="$2" config="$3" pretrained="$4" seed="$5" batch="$6"
  local output="${ROOT}/${label}"
  if [[ -s "${output}/summary.json" ]]; then
    echo "skip_completed=${label}"
    return
  fi
  [[ ! -e "$output" ]] || { echo "Refusing incomplete output: $output" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/train_spacenet8_opencd_smoke.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$config" "$output" \
    --model-name "$label" --pretrained-checkpoint "$pretrained" \
    --mode pre_post --device cuda:0 --epochs 40 --min-epochs 12 --patience 8 \
    --batch-size "$batch" --workers 4 --learning-rate 0.00006 \
    --positive-weight 5 --seed "$seed" >"${LOGS}/${label}.log" 2>&1
}

pids=()
for index in 0 1 2; do
  gpu=$((index + 1))
  seed=$((20260802 + index))
  run_one "$gpu" "changeformer_weight5_seed${seed}" "$CHANGE_CONFIG" "$MIT" "$seed" 3 &
  pids+=("$!")
done
for index in 0 1 2; do
  gpu=$((index + 4))
  [[ "$gpu" -eq 6 ]] && gpu=7
  seed=$((20260802 + index))
  run_one "$gpu" "ban_mit_weight5_seed${seed}" "$BAN_CONFIG" "$CLIP" "$seed" 1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
