#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
DATA="${STORE}/processed/spacenet8_germany/multi_post_spatial_block4_buffer1_v1_512"
OPENCD="${STORE}/external/open_cd"
CHANGE_CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BAN_CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
CHANGE_ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_block4_buffer1_v1_512"
BAN_ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_ban_block4_buffer1_v1_512"
OUTPUT="${STORE}/runs/spacenet8_germany/multi_post_spatial_block4_buffer1_v1_512"
LOGS="${STORE}/logs/spacenet8_multi_post_spatial_block4_buffer1_v1_512"

mkdir -p "$OUTPUT" "$LOGS"
while [[ ! -s "${DATA}/summary.json" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for multi-POST preparation"
  sleep 30
done
cd "$PROJECT"

run_one() {
  local gpu="$1" label="$2" config="$3" checkpoint="$4"
  local output="${OUTPUT}/${label}"
  if [[ -s "${output}/summary.json" ]]; then
    echo "skip_completed=${label}"
    return
  fi
  [[ ! -e "$output" ]] || { echo "Refusing incomplete output: $output" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/infer_spacenet8_multi_post_candidates.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$config" "$checkpoint" "$output" \
    --device cuda:0 --save-masks >"${LOGS}/${label}.log" 2>&1
}

pids=()
for index in 0 1 2; do
  gpu=$((index + 1))
  seed=$((20260802 + index))
  run_one "$gpu" "changeformer_weight5_seed${seed}" "$CHANGE_CONFIG" \
    "${CHANGE_ROOT}/changeformer_weight5_seed${seed}/best.pt" &
  pids+=("$!")
done
for index in 0 1 2; do
  gpu=$((index + 4))
  [[ "$gpu" -eq 6 ]] && gpu=7
  seed=$((20260802 + index))
  run_one "$gpu" "ban_mit_weight5_seed${seed}" "$BAN_CONFIG" \
    "${BAN_ROOT}/ban_mit_b0_local_weight5_seed${seed}/best.pt" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
exit "$status"
