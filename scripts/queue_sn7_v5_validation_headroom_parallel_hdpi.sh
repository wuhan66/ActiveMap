#!/usr/bin/env bash
set -euo pipefail

# This is a validation-only feasibility gate. It does not train a controller or
# select a checkpoint; it parallelizes independent oracle inference and records
# the exact aggregation receipt before the operation-wise headroom audit.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-/home/wh/ActiveMap/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
RUN_ROOT="${SN7_V5_RUN:-/home/wh/ActiveMap/runs/updater/v5_temporal_pair_trainval_r1_seed20260816}"
CHECKPOINT="${SN7_V5_HEADROOM_CHECKPOINT:-$RUN_ROOT/best_tradeoff.pt}"
EPISODES="$DATA_ROOT/episodes_trainval_v5.jsonl"
EPISODE_AUDIT="$DATA_ROOT/episodes_trainval_v5.audit.json"
SHARD_DIR="${SN7_V5_HEADROOM_SHARD_DIR:-$DATA_ROOT/headroom_val_v5_shards}"
STATES="${SN7_V5_HEADROOM_STATES:-$DATA_ROOT/selector_states_headroom_val_v5.jsonl}"
HEADROOM_DIR="${SN7_V5_HEADROOM_DIR:-$DATA_ROOT/nonkeep_candidate_headroom_val_v5}"
LOG_PREFIX="${SN7_V5_HEADROOM_LOG_PREFIX:-v5a}"
BOOTSTRAP_SEED="${SN7_V5_HEADROOM_BOOTSTRAP_SEED:-20260816}"
IFS=',' read -r -a GPUS <<< "${SN7_V5_HEADROOM_GPUS:-0,1,2,3}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ActiveMap Python runtime not found: $PYTHON" >&2
  exit 1
fi
if [[ ! -f "$EPISODES" || ! -f "$EPISODE_AUDIT" ]]; then
  echo "V5 episodes and their audit receipt must complete before headroom inference" >&2
  exit 1
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "V5 headroom checkpoint is missing: $CHECKPOINT" >&2
  exit 1
fi
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "at least one GPU is required" >&2
  exit 1
fi
if [[ -e "$SHARD_DIR" || -e "$STATES" || -e "$HEADROOM_DIR" ]]; then
  echo "Refusing to overwrite a V5 validation-headroom artifact" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$PROJECT_ROOT/scripts/shard_validation_selector_episodes.py" \
  "$EPISODES" "$SHARD_DIR" --shards "${#GPUS[@]}"

"$PYTHON" - "$CHECKPOINT" "$SHARD_DIR/checkpoint_receipt.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
receipt = Path(sys.argv[2])
digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
receipt.write_text(
    json.dumps(
        {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest,
            "selection": "explicit queue input",
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

pids=()
for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  episode_shard="$SHARD_DIR/episodes_val_shard_$(printf '%02d' "$index").jsonl"
  state_shard="$SHARD_DIR/selector_states_val_shard_$(printf '%02d' "$index").jsonl"
  log_path="$DATA_ROOT/selector_oracle_${LOG_PREFIX}_shard_$(printf '%02d' "$index").log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m activemap.cli build-selector-oracle \
    "$CHECKPOINT" "$episode_shard" "$state_shard" \
    --device cuda --image-size 128 --utility-mode executable --utility-profile balanced \
    --cost-weight 0.18 --false-edit-weight 0.35 --budgets 1.5,3.0,4.5 \
    --initial-evidence-strategy min_cost --splits val >"$log_path" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
if [[ "$status" -ne 0 ]]; then
  echo "at least one V5 validation oracle shard failed; partial shard artifacts retained" >&2
  exit "$status"
fi

"$PYTHON" "$PROJECT_ROOT/scripts/combine_validation_selector_states.py" \
  "$SHARD_DIR" "$STATES" --shards "${#GPUS[@]}"
"$PYTHON" "$PROJECT_ROOT/scripts/audit_nonkeep_candidate_headroom.py" \
  "$STATES" "$HEADROOM_DIR" --split val --minimum-rows 20 --minimum-aois 4 \
  --headroom-epsilon 0.000001 --bootstrap-draws 5000 --bootstrap-seed "$BOOTSTRAP_SEED"
