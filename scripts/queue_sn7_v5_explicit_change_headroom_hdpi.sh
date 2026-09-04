#!/usr/bin/env bash
set -euo pipefail

# This queue is deliberately separate from V5-A: no A artifact is overwritten.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-/home/wh/ActiveMap/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
RUN_ROOT="${SN7_V5B_RUN:-/home/wh/ActiveMap/runs/updater/v5_temporal_explicit_change_trainval_r1_seed20260817}"
MANIFEST="$DATA_ROOT/sn7_trainval_source.parquet"
EPISODES="${SN7_V5B_EPISODES:-$DATA_ROOT/episodes_trainval_v5.jsonl}"
EPISODE_AUDIT="${SN7_V5B_EPISODE_AUDIT:-$DATA_ROOT/episodes_trainval_v5.audit.json}"
STATES="$DATA_ROOT/selector_states_headroom_val_v5b.jsonl"
HEADROOM_DIR="$DATA_ROOT/nonkeep_candidate_headroom_val_v5b"
SHARD_DIR="$DATA_ROOT/headroom_val_v5b_shards"
GPUS="${SN7_V5B_HEADROOM_GPUS:-0,1,2,3}"
POLL_SECONDS="${SN7_V5_POLL_SECONDS:-60}"
TIMEOUT_SECONDS="${SN7_V5_TIMEOUT_SECONDS:-28800}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ActiveMap Python runtime not found: $PYTHON" >&2
  exit 1
fi
if [[ -e "$SHARD_DIR" || -e "$STATES" || -e "$HEADROOM_DIR" ]]; then
  echo "Refusing to overwrite an existing V5-B state or headroom artifact" >&2
  exit 1
fi
if [[ ! -f "$EPISODES" || ! -f "$EPISODE_AUDIT" ]]; then
  echo "V5-B requires the completed, audited immutable V5-A episode manifest" >&2
  exit 1
fi

started="$(date +%s)"
while [[ ! -f "$RUN_ROOT/metrics.json" ]]; do
  if (( $(date +%s) - started >= TIMEOUT_SECONDS )); then
    echo "Timed out waiting for V5-B updater metrics" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

CHECKPOINT="$RUN_ROOT/best_quality.pt"
if [[ ! -f "$CHECKPOINT" || ! -f "$MANIFEST" ]]; then
  echo "V5-B checkpoint or train/validation manifest missing" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" - "$DATA_ROOT/sn7_trainval_source.summary.json" "$RUN_ROOT/metrics.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path, metrics_path = (Path(value) for value in sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if manifest.get("test_assets_read") is not False:
    raise SystemExit("V5-B headroom requires a train/validation-only manifest receipt")
if metrics.get("stopped_by_user") is True:
    raise SystemExit("V5-B updater stopped by user; refusing headroom expansion")
if int(metrics.get("epochs_completed", 0)) < 1:
    raise SystemExit("V5-B updater has no completed epoch")
print(json.dumps({"status": "v5b_checkpoint_gate_passed", "epochs": metrics["epochs_completed"]}))
PY

SN7_V5_RUN="$RUN_ROOT" \
SN7_V5_HEADROOM_CHECKPOINT="$CHECKPOINT" \
SN7_V5_HEADROOM_SHARD_DIR="$SHARD_DIR" \
SN7_V5_HEADROOM_STATES="$STATES" \
SN7_V5_HEADROOM_DIR="$HEADROOM_DIR" \
SN7_V5_HEADROOM_LOG_PREFIX="v5b" \
SN7_V5_HEADROOM_BOOTSTRAP_SEED="20260817" \
SN7_V5_HEADROOM_GPUS="$GPUS" \
bash "$PROJECT_ROOT/scripts/queue_sn7_v5_validation_headroom_parallel_hdpi.sh"
