#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-gis/bin/python"
SOURCE="${STORE}/processed/spacenet8_germany/full202_v1/episodes.jsonl"

cd "$PROJECT"
pids=()
for rank in 2 3 100; do
  split="${STORE}/processed/spacenet8_germany/aligned_full202_spatial_active_rank${rank}_block4_buffer1_512"
  output="${STORE}/processed/spacenet8_germany/multi_post_spatial_active_rank${rank}_block4_buffer1_512"
  log="${STORE}/logs/prepare_spacenet8_multi_post_rank${rank}.log"
  if [[ ! -s "${output}/summary.json" ]]; then
    PYTHONPATH=src:. "$PYTHON" scripts/prepare_spacenet8_multi_post_candidates.py \
      "$SOURCE" "${split}/manifest.jsonl" "$output" --size 512 \
      >"$log" 2>&1 &
    pids+=("$!")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
