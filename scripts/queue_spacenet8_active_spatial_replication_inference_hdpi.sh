#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-opencd/bin/python"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
OPENCD="${STORE}/external/open_cd"
CONFIG="${OPENCD}/configs/changeformer/changeformer_mit-b0_256x256_40k_levircd.py"
TRAIN="${STORE}/runs/spacenet8_germany/opencd_spatial_active_replication_512"
OUTPUT="${STORE}/runs/spacenet8_germany/active_multi_post_spatial_replication_512"
LOGS="${STORE}/logs/spacenet8_active_multi_post_spatial_replication"

mkdir -p "$OUTPUT" "$LOGS"
cd "$PROJECT"
run_one() {
  local gpu="$1" rank="$2" seed="$3"
  local data="${STORE}/processed/spacenet8_germany/multi_post_spatial_active_rank${rank}_block4_buffer1_512"
  local train_label="rank${rank}_changeformer_weight5_seed${seed}"
  local label="${train_label}"
  local output="${OUTPUT}/${label}"
  while [[ ! -s "${data}/summary.json" || ! -s "${TRAIN}/${train_label}/summary.json" ]]; do sleep 20; done
  [[ ! -s "${output}/summary.json" ]] || return
  [[ ! -e "$output" ]] || return
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH=src:. "$PYTHON" \
    scripts/infer_spacenet8_multi_post_candidates.py \
    "${data}/manifest.jsonl" "$OPENCD" "$CONFIG" \
    "${TRAIN}/${train_label}/best.pt" "$output" --device cuda:0 --save-masks \
    >"${LOGS}/${label}.log" 2>&1
}

pids=()
run_one 1 10 20260802 & pids+=("$!")
run_one 2 10 20260803 & pids+=("$!")
run_one 3 10 20260804 & pids+=("$!")
run_one 4 50 20260802 & pids+=("$!")
run_one 5 50 20260803 & pids+=("$!")
run_one 7 50 20260804 & pids+=("$!")
for pid in "${pids[@]}"; do wait "$pid"; done

for rank in 10 50; do
  result="${OUTPUT}/rank${rank}_selector_safe_commit"
  if [[ ! -s "${result}/summary.json" ]]; then
    PYTHONPATH=src:. "$GIS_PYTHON" scripts/evaluate_spacenet8_active_multi_post.py \
      "$result" \
      "${OUTPUT}/rank${rank}_changeformer_weight5_seed20260802/per_candidate.jsonl" \
      "${OUTPUT}/rank${rank}_changeformer_weight5_seed20260803/per_candidate.jsonl" \
      "${OUTPUT}/rank${rank}_changeformer_weight5_seed20260804/per_candidate.jsonl" \
      >"${LOGS}/rank${rank}_selector_safe_commit.log" 2>&1
  fi
done
