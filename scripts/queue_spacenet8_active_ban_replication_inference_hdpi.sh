#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
OPENCD="${STORE}/external/open_cd"
CONFIG="${PROJECT}/configs/opencd/ban_vit_b16_clip_mit_b0_local_hdpi.py"
TRAIN="${STORE}/runs/spacenet8_germany/opencd_spatial_active_replication_ban_512"
OUTPUT="${STORE}/runs/spacenet8_germany/active_multi_post_spatial_replication_ban_512"
LOGS="${STORE}/logs/spacenet8_active_multi_post_spatial_replication_ban"

mkdir -p "$OUTPUT" "$LOGS"
cd "$PROJECT"

run_one() {
  local gpu="$1" rank="$2" seed="$3"
  local data="${STORE}/processed/spacenet8_germany/multi_post_spatial_active_rank${rank}_block4_buffer1_512"
  local label="rank${rank}_ban_mit_weight5_seed${seed}"
  local output="${OUTPUT}/${label}"
  while [[ ! -s "${data}/summary.json" || ! -s "${TRAIN}/${label}/summary.json" ]]; do sleep 20; done
  [[ ! -s "${output}/summary.json" ]] || return
  [[ ! -e "$output" ]] || return
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/infer_spacenet8_multi_post_candidates.py \
    "${data}/manifest.jsonl" "$OPENCD" "$CONFIG" \
    "${TRAIN}/${label}/best.pt" "$output" --device cuda:0 --save-masks \
    >"${LOGS}/${label}.log" 2>&1
}

pids=()
run_one 1 10 20260802 & pids+=("$!")
run_one 2 10 20260803 & pids+=("$!")
run_one 3 10 20260804 & pids+=("$!")
run_one 1 50 20260802 & pids+=("$!")
run_one 2 50 20260803 & pids+=("$!")
run_one 3 50 20260804 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

for rank in 10 50; do
  result="${OUTPUT}/rank${rank}_selector_safe_commit"
  if [[ ! -s "${result}/summary.json" ]]; then
    PYTHONPATH=src:. "$GIS_PYTHON" scripts/evaluate_spacenet8_active_multi_post.py \
      "$result" \
      "${OUTPUT}/rank${rank}_ban_mit_weight5_seed20260802/per_candidate.jsonl" \
      "${OUTPUT}/rank${rank}_ban_mit_weight5_seed20260803/per_candidate.jsonl" \
      "${OUTPUT}/rank${rank}_ban_mit_weight5_seed20260804/per_candidate.jsonl" \
      >"${LOGS}/rank${rank}_selector_safe_commit.log" 2>&1
  fi
  bootstrap="${OUTPUT}/rank${rank}_bootstrap5000"
  if [[ ! -s "${bootstrap}/summary.json" ]]; then
    PYTHONPATH=src:. "$GIS_PYTHON" scripts/bootstrap_spacenet8_active_multi_post.py \
      "$bootstrap" \
      "${OUTPUT}/rank${rank}_ban_mit_weight5_seed20260802/per_candidate.jsonl" \
      "${OUTPUT}/rank${rank}_ban_mit_weight5_seed20260803/per_candidate.jsonl" \
      "${OUTPUT}/rank${rank}_ban_mit_weight5_seed20260804/per_candidate.jsonl" \
      --draws 5000 >"${LOGS}/rank${rank}_bootstrap5000.log" 2>&1
  fi
done
