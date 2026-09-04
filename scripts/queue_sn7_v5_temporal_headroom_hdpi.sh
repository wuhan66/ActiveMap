#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-/home/wh/ActiveMap/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
RUN_ROOT="${SN7_V5_RUN:-/home/wh/ActiveMap/runs/updater/v5_temporal_pair_trainval_r1_seed20260816}"
MANIFEST="$DATA_ROOT/sn7_trainval_source.parquet"
EPISODES="$DATA_ROOT/episodes_trainval_v5.jsonl"
STATES="$DATA_ROOT/selector_states_trainval_v5.jsonl"
HEADROOM_DIR="$DATA_ROOT/nonkeep_candidate_headroom_val_v5"
POLL_SECONDS="${SN7_V5_POLL_SECONDS:-60}"
TIMEOUT_SECONDS="${SN7_V5_TIMEOUT_SECONDS:-28800}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ActiveMap Python runtime not found: $PYTHON" >&2
  exit 1
fi
if [[ -e "$EPISODES" || -e "$STATES" || -e "$HEADROOM_DIR" ]]; then
  echo "Refusing to overwrite an existing V5 headroom artifact" >&2
  exit 1
fi

started="$(date +%s)"
while [[ ! -f "$RUN_ROOT/metrics.json" ]]; do
  if (( $(date +%s) - started >= TIMEOUT_SECONDS )); then
    echo "Timed out waiting for V5 updater metrics" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

CHECKPOINT="$RUN_ROOT/best_tradeoff.pt"
if [[ ! -f "$CHECKPOINT" || ! -f "$MANIFEST" ]]; then
  echo "V5 checkpoint or train/validation manifest missing" >&2
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
    raise SystemExit("V5 headroom requires a train/validation-only manifest receipt")
if metrics.get("stopped_by_user") is True:
    raise SystemExit("V5 updater stopped by user; refusing headroom expansion")
if int(metrics.get("epochs_completed", 0)) < 1:
    raise SystemExit("V5 updater has no completed epoch")
print(json.dumps({"status": "v5_checkpoint_gate_passed", "epochs": metrics["epochs_completed"]}))
PY

"$PYTHON" -m activemap.cli build-episodes-sn7 "$MANIFEST" "$EPISODES" \
  --max-per-operation 20 --max-month-gap 1 --min-change-persistence 2 \
  --min-area 16 --max-centroid-distance 20 --seed 20260710 \
  --contexts 0,32,96 --scales 1,2,4
"$PYTHON" -m activemap.cli audit-episodes "$EPISODES" \
  "$DATA_ROOT/episodes_trainval_v5.audit.json" \
  --expected-derivation-version sn7-adjacent-v3-distance-gated
"$PYTHON" -m activemap.cli build-selector-oracle "$CHECKPOINT" "$EPISODES" "$STATES" \
  --device auto --image-size 128 --utility-mode executable --utility-profile balanced \
  --cost-weight 0.18 --false-edit-weight 0.35 --budgets 1.5,3.0,4.5 \
  --initial-evidence-strategy min_cost --splits train,val
"$PYTHON" "$PROJECT_ROOT/scripts/audit_nonkeep_candidate_headroom.py" \
  "$STATES" "$HEADROOM_DIR" --split val --minimum-rows 20 --minimum-aois 4 \
  --headroom-epsilon 0.000001 --bootstrap-draws 5000 --bootstrap-seed 20260816
