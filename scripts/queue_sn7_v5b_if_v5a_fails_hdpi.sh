#!/usr/bin/env bash
set -euo pipefail

# Start V5-B only after the predeclared V5-A oracle headroom gate rejects A.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-/home/wh/ActiveMap/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
A_HEADROOM="$DATA_ROOT/nonkeep_candidate_headroom_val_v5/summary.json"
SUPPORT_RECEIPT="$DATA_ROOT/temporal_supervision_support_v5.json"
B_RUN="${SN7_V5B_RUN:-/home/wh/ActiveMap/runs/updater/v5_temporal_explicit_change_trainval_r1_seed20260817}"
B_CONFIG="${SN7_V5B_CONFIG:-$PROJECT_ROOT/configs/updater/sn7_v5_temporal_explicit_change_seed20260817_server.yaml}"
B_HEADROOM="$DATA_ROOT/nonkeep_candidate_headroom_val_v5b/summary.json"
STATUS="$DATA_ROOT/v5b_after_v5a_gate.json"
GPU="${SN7_V5B_GPU:-0}"
POLL_SECONDS="${SN7_V5_POLL_SECONDS:-60}"
TIMEOUT_SECONDS="${SN7_V5_TIMEOUT_SECONDS:-43200}"
LOG_DIR="/home/wh/ActiveMap/logs"

if [[ ! -x "$PYTHON" || ! -f "$B_CONFIG" ]]; then
  echo "V5-B runtime or configuration missing" >&2
  exit 1
fi
if [[ -e "$STATUS" || -d "$B_RUN" || -e "$B_HEADROOM" ]]; then
  echo "V5-B gate or output already exists; refusing duplicate launch" >&2
  exit 1
fi

started="$(date +%s)"
while [[ ! -f "$A_HEADROOM" ]]; do
  if (( $(date +%s) - started >= TIMEOUT_SECONDS )); then
    echo "Timed out waiting for V5-A headroom summary" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
if [[ ! -f "$SUPPORT_RECEIPT" ]]; then
  "$PYTHON" "$PROJECT_ROOT/scripts/audit_temporal_supervision_support.py" \
    "$DATA_ROOT/updater_samples.jsonl" "$SUPPORT_RECEIPT" --allowed-splits train,val
fi

"$PYTHON" - "$A_HEADROOM" "$SUPPORT_RECEIPT" "$STATUS" <<'PY'
import json
import sys
from pathlib import Path

headroom_path, support_path, status_path = (Path(value) for value in sys.argv[1:])
headroom = json.loads(headroom_path.read_text(encoding="utf-8"))
support = json.loads(support_path.read_text(encoding="utf-8"))
if support.get("test_assets_read") is not False or support.get("status") != "passed":
    raise SystemExit("temporal support receipt is not train/validation-only and passed")
passed = bool(headroom.get("gate", {}).get("passes_candidate_recovery_preflight"))
status = {
    "schema_version": "sn7-v5b-after-v5a-gate-v1",
    "v5a_headroom_summary": str(headroom_path.resolve()),
    "temporal_support_receipt": str(support_path.resolve()),
    "v5a_passed_candidate_recovery_preflight": passed,
    "test_assets_read": False,
}
status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
print("pass" if passed else "fail")
PY

if "$PYTHON" - "$STATUS" <<'PY'
import json
import sys

raise SystemExit(0 if json.load(open(sys.argv[1]))["v5a_passed_candidate_recovery_preflight"] else 1)
PY
then
  echo "V5-A passed all non-KEEP headroom slices; V5-B is intentionally skipped."
  exit 0
fi

mkdir -p "$LOG_DIR"
nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$PYTHONPATH" "$PYTHON" -m activemap.cli train-updater "$B_CONFIG" \
  >"$LOG_DIR/train_sn7_v5_temporal_explicit_change_seed20260817.log" 2>&1 &
nohup flock -n "$LOG_DIR/queue_sn7_v5_explicit_change_headroom_hdpi.lock" \
  bash "$PROJECT_ROOT/scripts/queue_sn7_v5_explicit_change_headroom_hdpi.sh" \
  >"$LOG_DIR/queue_sn7_v5_explicit_change_headroom_hdpi_20260817.log" 2>&1 &
echo "V5-A failed the non-KEEP headroom gate; launched isolated V5-B on GPU $GPU."
