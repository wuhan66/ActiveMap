#!/usr/bin/env bash
set -euo pipefail

# Export literal V5 validation tables only after the independent intake audit.
# This sidecar never reads test data or writes into the ICLR manuscript tree.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${V5_MATCHED_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_matched_nonkeep_2x2_v2}"
AUDIT="${RUN_ROOT}/v5_matched_intake_audit.json"
AGGREGATE="${RUN_ROOT}/three_seed_nonkeep_factorial_with_forced_summary.json"
OUTPUT_ROOT="${RUN_ROOT}/generated_tex"
STATUS_PATH="${RUN_ROOT}/v5_tex_export_status.json"
LOG_DIR="${STORAGE_ROOT}/logs"

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ -d "${RUN_ROOT}" ]] || { echo "V5 matched run root is missing: ${RUN_ROOT}" >&2; exit 1; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "V5 TeX output already exists: ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec 9>"${RUN_ROOT}/.v5_tex_export.lock"
flock -n 9 || { echo "V5 TeX export sidecar is already active" >&2; exit 1; }

write_status() {
  "${PYTHON}" - "${STATUS_PATH}" "$1" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"status": sys.argv[2], "split": "val", "test_assets_read": False}, indent=2)
    + "\n",
    encoding="utf-8",
)
PY
}

write_status "waiting_for_intake_audit"
while [[ ! -f "${AUDIT}" ]]; do sleep 120; done

"${PYTHON}" - "${AUDIT}" "${AGGREGATE}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

audit_path, aggregate_path = map(Path, sys.argv[1:])
audit = json.loads(audit_path.read_text(encoding="utf-8"))
if audit.get("passed") is not True or audit.get("split") != "val" or audit.get("test_assets_read") is not False:
    raise SystemExit("V5 intake audit is not a passed validation-only receipt")
aggregate = audit.get("aggregate", {})
if aggregate.get("path") != str(aggregate_path.resolve()):
    raise SystemExit("V5 intake audit does not bind the expected aggregate path")
actual = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
if aggregate.get("sha256") != actual:
    raise SystemExit("V5 intake audit aggregate hash mismatch")
PY

write_status "generating_validation_tables"
"${PYTHON}" "${PROJECT_ROOT}/scripts/generate_sn7_v5_factorial_tex.py" \
  "${AGGREGATE}" "${OUTPUT_ROOT}" \
  >"${LOG_DIR}/sn7_v5_factorial_tex_export.log" 2>&1

"${PYTHON}" - "${AUDIT}" "${AGGREGATE}" "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

audit_path, aggregate_path, output_root = map(Path, sys.argv[1:])
manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
aggregate_hash = hashlib.sha256(aggregate_path.read_bytes()).hexdigest()
if manifest.get("split") != "val" or manifest.get("test_assets_read") is not False:
    raise SystemExit("generated V5 table manifest is not validation-only")
if manifest.get("source_summary") != str(aggregate_path.resolve()):
    raise SystemExit("generated V5 table manifest has an unexpected source")
if manifest.get("source_summary_sha256") != aggregate_hash:
    raise SystemExit("generated V5 table manifest source hash mismatch")
receipt = {
    "schema_version": "sn7-v5-factorial-tex-export-receipt-v1",
    "split": "val",
    "test_assets_read": False,
    "audit": {
        "path": str(audit_path.resolve()),
        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
    },
    "aggregate": {
        "path": str(aggregate_path.resolve()),
        "sha256": aggregate_hash,
    },
    "generated_manifest": str((output_root / "manifest.json").resolve()),
    "promotion": manifest.get("promotion"),
}
(output_root / "v5_factorial_tex_export_receipt.json").write_text(
    json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
)
PY

write_status "complete"
echo "V5 validation-only TeX export complete: ${OUTPUT_ROOT}"
