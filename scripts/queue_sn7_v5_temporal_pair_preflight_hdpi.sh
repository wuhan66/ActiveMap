#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-/home/wh/ActiveMap/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
RUN_ROOT="${SN7_V5_RUN:-/home/wh/ActiveMap/runs/updater/v5_temporal_pair_trainval_r1_seed20260816}"
POLL_SECONDS="${SN7_V5_POLL_SECONDS:-60}"
TIMEOUT_SECONDS="${SN7_V5_TIMEOUT_SECONDS:-14400}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ActiveMap Python runtime not found: $PYTHON" >&2
  exit 1
fi
if [[ -e "$RUN_ROOT/metrics.json" || -e "$RUN_ROOT/last.pt" ]]; then
  echo "Refusing to overwrite an existing V5 preflight run: $RUN_ROOT" >&2
  exit 1
fi

started="$(date +%s)"
while [[ ! -f "$DATA_ROOT/summary.json" || ! -f "$DATA_ROOT/audit.json" || ! -f "$DATA_ROOT/qc_train_val_v5/index.json" ]]; do
  if (( $(date +%s) - started >= TIMEOUT_SECONDS )); then
    echo "Timed out waiting for V5 data, audit, and QC receipts" >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

"$PYTHON" - "$DATA_ROOT/audit.json" "$DATA_ROOT/qc_train_val_v5/index.json" <<'PY'
import json
import sys
from pathlib import Path

audit_path, qc_path = (Path(value) for value in sys.argv[1:])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
qc = json.loads(qc_path.read_text(encoding="utf-8"))
if audit.get("passed") is not True:
    raise SystemExit("V5 updater audit did not pass; refusing GPU training")
if qc.get("test_assets_rendered") is True:
    raise SystemExit("V5 QC rendered test assets; refusing GPU training")
if set(qc.get("allowed_splits") or []) - {"train", "val"}:
    raise SystemExit("V5 QC includes a split outside train/val")
if int(qc.get("rendered", 0)) < 1:
    raise SystemExit("V5 QC did not render any samples")
print(json.dumps({"status": "v5_data_gate_passed", "audit": str(audit_path), "qc": str(qc_path)}))
PY

exec bash "$PROJECT_ROOT/scripts/run_sn7_v5_temporal_pair_preflight_hdpi.sh"
