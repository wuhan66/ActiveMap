#!/usr/bin/env bash
set -euo pipefail

# Wait for the registered forced-control sidecar and then produce exactly one
# fail-closed V5 intake receipt. This process does not run a model or read test
# data.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${V5_MATCHED_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_matched_nonkeep_2x2_v2}"
STATUS_PATH="${RUN_ROOT}/v5_intake_audit_status.json"
FORCED_STATUS_PATH="${RUN_ROOT}/forced_cost_control_status.json"
OUTPUT="${RUN_ROOT}/v5_matched_intake_audit.json"

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ -d "${RUN_ROOT}" ]] || { echo "V5 matched run root is missing: ${RUN_ROOT}" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

write_status() {
  "${PYTHON}" - "${STATUS_PATH}" "$1" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"status": sys.argv[2], "split": "train,val", "test_assets_read": False}, indent=2)
    + "\n",
    encoding="utf-8",
)
PY
}

write_status "waiting_for_forced_control"
while true; do
  [[ -f "${FORCED_STATUS_PATH}" ]] || { sleep 120; continue; }
  forced_status="$("${PYTHON}" - "${FORCED_STATUS_PATH}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
PY
)"
  [[ "${forced_status}" == "complete" ]] && break
  sleep 120
done

[[ ! -e "${OUTPUT}" ]] || { echo "V5 intake audit already exists" >&2; exit 1; }
write_status "auditing"
"${PYTHON}" "${PROJECT_ROOT}/scripts/audit_sn7_v5_matched_intake.py" \
  "${RUN_ROOT}" "${OUTPUT}"
write_status "complete"
echo "V5 matched intake audit complete: ${OUTPUT}"
