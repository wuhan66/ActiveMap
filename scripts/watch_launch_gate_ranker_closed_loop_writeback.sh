#!/usr/bin/env bash
set -euo pipefail

ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${ACTIVEMAP_REPOSITORY:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-$ROOT/envs/activemap-agent/bin/python}"
GPU="${GATE_RANKER_CLOSED_LOOP_GPU:-0}"
POLL_SECONDS="${POLL_SECONDS:-180}"
DECISION="${GATE_RANKER_DECISION:-$ROOT/artifacts/sn7_active_catalog/gate_ranker_round_20260720/promotion_decision.json}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
RANKER="${RANKER:-$ROOT/runs/sn7_active_catalog/candidate_ranker_v4/seed20260720/best.pt}"
DATA_ROOT="${DATA_ROOT:-$ROOT/processed/sn7_v1/agent/sequential_selector_v1}"
STATES="${STATES:-$DATA_ROOT/selector_states_train_val.jsonl}"
EPISODES="${EPISODES:-$DATA_ROOT/full/episodes_train_val.jsonl}"
SFT_ROOT="${SFT_ROOT:-$DATA_ROOT/full/active_catalog_sft_v4}"
BUNDLE_ROOT="${BUNDLE_ROOT:-$DATA_ROOT/full/closed_loop_v1}"
UPDATER="${UPDATER:-$ROOT/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt}"
OUTPUT="${OUTPUT:-$ROOT/runs/sn7_active_catalog/promoted_gate_ranker_closed_loop_v1}"
BASELINE_OUTPUT="${BASELINE_OUTPUT:-$ROOT/runs/sn7_active_catalog/promoted_gate_ranker_baseline_matrix_v1}"
ASSET_ROOT_MAP="${SN7_ASSET_ROOT_MAP:-/mnt/mydisk/wh/ActiveMap/datasets/sn7=$ROOT/datasets/sn7}"

cd "$REPO"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "$GPU"

while [[ ! -s "$DECISION" ]]; do
  echo "$(date --iso-8601=seconds) waiting for Gate+Ranker promotion decision"
  sleep "$POLL_SECONDS"
done

passed="$($PYTHON -c '
import json, sys
row = json.load(open(sys.argv[1]))
assert row.get("test_assets_read") is False
print(int(row.get("promotion", {}).get("passed") is True))
' "$DECISION")"
if [[ "$passed" != "1" ]]; then
  echo "Gate+Ranker did not pass component promotion; closed-loop launch withheld" >&2
  exit 3
fi

while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "$(date --iso-8601=seconds) waiting for GPU $GPU"
  sleep "$POLL_SECONDS"
done

pipeline_completed() {
  [[ -s "$1" ]] && "$PYTHON" -c '
import json, sys
raise SystemExit(0 if json.load(open(sys.argv[1])).get("status") == "completed" else 1)
' "$1"
}

if ! pipeline_completed "$OUTPUT/pipeline_state.json"; then
  [[ ! -e "$OUTPUT" ]] || { echo "incomplete candidate output: $OUTPUT" >&2; exit 4; }
  "$PYTHON" scripts/launch_gate_ranker_closed_loop_writeback.py \
    "$DECISION" "$MODEL" "$RANKER" "$STATES" "$EPISODES" \
    "$SFT_ROOT/val.jsonl" "$SFT_ROOT/val_evaluation_index.jsonl" \
    "$UPDATER" "$BUNDLE_ROOT" "$OUTPUT" \
    --gpu "$GPU" --python "$PYTHON" --seed 20260720 \
    --max-candidates 16 --max-acquisitions 2 --bootstrap-repetitions 2000 \
    --image-size 512 --threshold 0.5 --asset-root-map "$ASSET_ROOT_MAP"
fi

if ! pipeline_completed "$BASELINE_OUTPUT/pipeline_state.json"; then
  [[ ! -e "$BASELINE_OUTPUT" ]] || {
    echo "incomplete baseline output: $BASELINE_OUTPUT" >&2
    exit 5
  }
  exec "$PYTHON" scripts/launch_gate_ranker_baseline_matrix.py \
    "$BUNDLE_ROOT/states_val_step0.jsonl" "$BUNDLE_ROOT/episodes_val.jsonl" \
    "$RANKER" "$UPDATER" "$OUTPUT/closed_loop/evaluation/traces.jsonl" \
    "$OUTPUT/writeback/evaluation/writeback.jsonl" "$BASELINE_OUTPUT" \
    --gpu "$GPU" --python "$PYTHON" --seed 20260720 \
    --max-candidates 16 --max-acquisitions 2 --bootstrap-repetitions 2000 \
    --image-size 512 --threshold 0.5 --asset-root-map "$ASSET_ROOT_MAP"
fi
