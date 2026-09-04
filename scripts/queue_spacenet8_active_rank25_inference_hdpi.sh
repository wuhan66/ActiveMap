#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
OPENCD="${STORE}/external/open_cd"
CHANGE_CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
BAN_CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
DATA="${STORE}/processed/spacenet8_germany/multi_post_spatial_active_rank25_block4_buffer1_512"
TRAIN="${STORE}/runs/spacenet8_germany/opencd_spatial_active_rank25_512"
OUTPUT="${STORE}/runs/spacenet8_germany/active_multi_post_spatial_rank25_512"
LOGS="${STORE}/logs/spacenet8_active_multi_post_rank25"

mkdir -p "$OUTPUT" "$LOGS"
cd "$PROJECT"

run_one() {
  local gpu="$1" backend="$2" seed="$3" config
  if [[ "$backend" == changeformer ]]; then config="$CHANGE_CONFIG"; else config="$BAN_CONFIG"; fi
  local label="${backend}_weight5_seed${seed}"
  local output="${OUTPUT}/${label}"
  while [[ ! -s "${DATA}/summary.json" || ! -s "${TRAIN}/${label}/summary.json" ]]; do sleep 20; done
  [[ ! -s "${output}/summary.json" ]] || return
  [[ ! -e "$output" ]] || return
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/infer_spacenet8_multi_post_candidates.py \
    "${DATA}/manifest.jsonl" "$OPENCD" "$config" \
    "${TRAIN}/${label}/best.pt" "$output" --device cuda:0 --save-masks \
    >"${LOGS}/${label}.log" 2>&1
}

pids=()
run_one 1 changeformer 20260802 & pids+=("$!")
run_one 2 changeformer 20260803 & pids+=("$!")
run_one 3 changeformer 20260804 & pids+=("$!")
run_one 4 ban_mit 20260802 & pids+=("$!")
run_one 5 ban_mit 20260803 & pids+=("$!")
run_one 7 ban_mit 20260804 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

for backend in changeformer ban_mit; do
  result="${OUTPUT}/${backend}_selector_safe_commit"
  inputs=(
    "${OUTPUT}/${backend}_weight5_seed20260802/per_candidate.jsonl"
    "${OUTPUT}/${backend}_weight5_seed20260803/per_candidate.jsonl"
    "${OUTPUT}/${backend}_weight5_seed20260804/per_candidate.jsonl"
  )
  if [[ ! -s "${result}/summary.json" ]]; then
    PYTHONPATH=src:. "$GIS_PYTHON" scripts/evaluate_spacenet8_active_multi_post.py \
      "$result" "${inputs[@]}" >"${LOGS}/${backend}_selector_safe_commit.log" 2>&1
  fi
  bootstrap="${OUTPUT}/${backend}_bootstrap5000"
  if [[ ! -s "${bootstrap}/summary.json" ]]; then
    PYTHONPATH=src:. "$GIS_PYTHON" scripts/bootstrap_spacenet8_active_multi_post.py \
      "$bootstrap" "${inputs[@]}" --draws 5000 \
      >"${LOGS}/${backend}_bootstrap5000.log" 2>&1
  fi
done
