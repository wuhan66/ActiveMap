#!/usr/bin/env bash
set -euo pipefail

# Wait for the registered V5 intake audit, then export only the nine
# predeclared validation cases into a supplementary qualitative asset pack.
# This sidecar never opens test data and never produces a main-paper figure.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
EPISODES="${DATA_ROOT}/episodes_trainval_v5.jsonl"
RUN_ROOT="${V5_MATCHED_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_matched_nonkeep_2x2_v2}"
SEED=20260817
AUDIT="${RUN_ROOT}/v5_matched_intake_audit.json"
AGGREGATE="${RUN_ROOT}/three_seed_nonkeep_factorial_with_forced_summary.json"
TRACE="${RUN_ROOT}/rollouts/seed${SEED}_val/selected_rollouts.jsonl"
EVIDENCE_ROOT="${RUN_ROOT}/qualitative_evidence_seed${SEED}"
OUTPUT_ROOT="${RUN_ROOT}/qualitative_supplement_seed${SEED}"
STATUS_PATH="${RUN_ROOT}/qualitative_export_status.json"
LOG_DIR="${STORAGE_ROOT}/logs"

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ -f "${EPISODES}" ]] || { echo "V5 episode manifest is missing: ${EPISODES}" >&2; exit 1; }
[[ -d "${RUN_ROOT}" ]] || { echo "V5 run root is missing: ${RUN_ROOT}" >&2; exit 1; }
[[ ! -e "${EVIDENCE_ROOT}" ]] || { echo "V5 evidence output already exists: ${EVIDENCE_ROOT}" >&2; exit 1; }
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "V5 qualitative output already exists: ${OUTPUT_ROOT}" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${LOG_DIR}"

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

[[ -f "${TRACE}" ]] || { echo "V5 selected rollout is missing: ${TRACE}" >&2; exit 1; }
write_status "exporting_fixed_evidence"
"${PYTHON}" "${PROJECT_ROOT}/scripts/figures/export_sn7_v5_qualitative_evidence.py" \
  "${RUN_ROOT}/qualitative_manifest.json" "${EPISODES}" "${TRACE}" "${EVIDENCE_ROOT}" \
  --image-size 384 >"${LOG_DIR}/sn7_v5_qualitative_evidence_seed${SEED}.log" 2>&1

write_status "rendering_supplement"
"${PYTHON}" "${PROJECT_ROOT}/scripts/figures/render_sn7_v5_matched_writeback.py" \
  "${RUN_ROOT}/qualitative_manifest.json" "${EPISODES}" "${RUN_ROOT}" "${OUTPUT_ROOT}" \
  --scope supplement --render-seed "${SEED}" --evidence-root "${EVIDENCE_ROOT}" \
  --aggregate-summary "${AGGREGATE}" --panel-size 384 \
  >"${LOG_DIR}/sn7_v5_qualitative_render_seed${SEED}.log" 2>&1

write_status "complete"
echo "V5 fixed validation qualitative supplement complete: ${OUTPUT_ROOT}"
