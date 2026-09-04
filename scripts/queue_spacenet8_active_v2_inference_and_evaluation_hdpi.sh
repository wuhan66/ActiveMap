#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
DATA="${STORE}/processed/spacenet8_germany/multi_post_spatial_active_block4_buffer1_v2_512"
OPENCD="${STORE}/external/open_cd"
CHANGE_CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BAN_CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
TRAIN_ROOT="${STORE}/runs/spacenet8_germany/opencd_spatial_active_block4_buffer1_v2_512"
OUTPUT="${STORE}/runs/spacenet8_germany/active_multi_post_spatial_v2_512"
LOGS="${STORE}/logs/spacenet8_active_multi_post_spatial_v2_512"

mkdir -p "$OUTPUT" "$LOGS"
cd "$PROJECT"

run_inference() {
  local gpu="$1" label="$2" config="$3"
  local output="${OUTPUT}/${label}"
  while [[ ! -s "${TRAIN_ROOT}/${label}/summary.json" ]]; do
    echo "[$(date --iso-8601=seconds)] waiting for ${label}"
    sleep 30
  done
  if [[ -s "${output}/summary.json" ]]; then
    return
  fi
  [[ ! -e "$output" ]] || { echo "Refusing incomplete output: $output" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/infer_spacenet8_multi_post_candidates.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$config" \
    "${TRAIN_ROOT}/${label}/best.pt" "$output" --device cuda:0 --save-masks \
    >"${LOGS}/${label}.log" 2>&1
}

pids=()
for index in 0 1 2; do
  seed=$((20260802 + index))
  run_inference "$((index + 1))" "changeformer_weight5_seed${seed}" "$CHANGE_CONFIG" &
  pids+=("$!")
done
for index in 0 1 2; do
  gpu=$((index + 4))
  [[ "$gpu" -eq 6 ]] && gpu=7
  seed=$((20260802 + index))
  run_inference "$gpu" "ban_mit_weight5_seed${seed}" "$BAN_CONFIG" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

for backend in changeformer ban_mit; do
  result="${OUTPUT}/${backend}_selector_safe_commit"
  if [[ ! -s "${result}/summary.json" ]]; then
    PYTHONPATH=src:. "$GIS_PYTHON" scripts/evaluate_spacenet8_active_multi_post.py \
      "$result" \
      "${OUTPUT}/${backend}_weight5_seed20260802/per_candidate.jsonl" \
      "${OUTPUT}/${backend}_weight5_seed20260803/per_candidate.jsonl" \
      "${OUTPUT}/${backend}_weight5_seed20260804/per_candidate.jsonl" \
      >"${LOGS}/${backend}_selector_safe_commit.log" 2>&1
  fi
done
