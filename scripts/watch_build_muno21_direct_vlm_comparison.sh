#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
V1="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_zeroshot_v1"
V2="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_operational_prompt_v2"
V3="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_fourshot_v3"
OUTPUT="${STORAGE_ROOT}/artifacts/paper_evidence/muno21_direct_vlm_prompt_comparison_v1"

mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0
for root in "${V1}" "${V2}" "${V3}"; do
  while [[ ! -s "${root}/COMPLETE.json" ]]; do
    [[ ! -s "${root}/FAILED.json" ]] || exit 4
    sleep 30
  done
done

cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/build_muno21_direct_vlm_comparison.py "${OUTPUT}" \
  --run "Minimal prompt=${V1}" \
  --run "Operational prompt=${V2}" \
  --run "Four-shot visual=${V3}"

OUTPUT="${OUTPUT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
files = {}
for name in ("table.json", "table.csv", "table.md"):
    path = root / name
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": "muno21-direct-vlm-comparison-complete-v1",
            "status": "complete",
            "files": files,
            "split": "val",
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
