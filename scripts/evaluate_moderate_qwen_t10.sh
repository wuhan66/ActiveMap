#!/usr/bin/env bash
set -euo pipefail

ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${ACTIVEMAP_REPOSITORY:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-$ROOT/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
SFT="${SFT_ROOT:-$ROOT/processed/sn7_v1/agent/sequential_selector_v1/full/active_catalog_sft_v4}"
RUN="${RUN_ROOT:-$ROOT/runs/sn7_active_catalog/qwen3vl4b_moderate_sampling_t10_seed1/seed20260717}"
SENTINEL="$SFT/balanced_sentinel_v1"
SELECTION_ROOT="$RUN/sentinel_checkpoint_selection"
SELECTION="$SELECTION_ROOT/selection.json"
FULL_OUTPUT="$RUN/active_catalog_val"

cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO"
mkdir -p "$SELECTION_ROOT"

evaluate_sentinel() {
  local label="$1" adapter="$2" output="$SELECTION_ROOT/$1"
  if [[ ! -s "$output/summary.json" ]]; then
    [[ ! -e "$output" ]] || { echo "incomplete output: $output" >&2; exit 4; }
    "$PYTHON" scripts/evaluate_active_catalog_selector.py \
      "$MODEL" "$adapter" "$SENTINEL/val.jsonl" \
      "$SENTINEL/val_evaluation_index.jsonl" "$output" \
      --device cuda:0 --seed 20260717 --bootstrap-repetitions 500
  fi
}

evaluate_sentinel checkpoint-6000 "$RUN/checkpoints/checkpoint-6000"
evaluate_sentinel final "$RUN/final"

if [[ ! -s "$SELECTION" ]]; then
  "$PYTHON" scripts/select_active_catalog_sentinel_checkpoint.py "$SELECTION" \
    --candidate checkpoint-6000 "$RUN/checkpoints/checkpoint-6000" \
      "$SELECTION_ROOT/checkpoint-6000/traces.jsonl" \
    --candidate final "$RUN/final" "$SELECTION_ROOT/final/traces.jsonl"
fi

selected="$($PYTHON -c \
  'import json,sys; r=json.load(open(sys.argv[1])); assert r["passed"] and not r["test_assets_read"]; print(r["selected"]["adapter"])' \
  "$SELECTION")"

if [[ ! -s "$FULL_OUTPUT/summary.json" ]]; then
  [[ ! -e "$FULL_OUTPUT" ]] || { echo "incomplete output: $FULL_OUTPUT" >&2; exit 5; }
  "$PYTHON" scripts/evaluate_active_catalog_selector.py \
    "$MODEL" "$selected" "$SFT/val.jsonl" "$SFT/val_evaluation_index.jsonl" \
    "$FULL_OUTPUT" --device cuda:0 --seed 20260717 --bootstrap-repetitions 2000
fi

echo "completed: $FULL_OUTPUT/summary.json"
