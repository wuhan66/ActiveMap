#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
AGGREGATE="${STORAGE_ROOT}/artifacts/paper_evidence/muno21_direct_vlm_supervised_2seed_v1"
OUTPUT="${AGGREGATE}/prompt_to_supervised_comparison"
LOG="${AGGREGATE}/comparison_watcher.log"

mkdir -p "${AGGREGATE}"
exec 8>"${AGGREGATE}/.comparison.lock"
flock -n 8 || exit 0
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0

while [[ ! -s "${AGGREGATE}/COMPLETE.json" ]]; do
  [[ ! -s "${AGGREGATE}/FAILED.json" ]] || exit 3
  sleep 60
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" scripts/build_muno21_direct_vlm_comparison.py \
  "${OUTPUT}" \
  --run "Minimal zero-shot=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_zeroshot_v1" \
  --run "Operational zero-shot=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_operational_prompt_v2" \
  --run "Four-shot visual ICL=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_fourshot_v3" \
  --run "Supervised Direct-VLM seed 20260831=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_sft_v1_seed20260831" \
  --run "Supervised Direct-VLM seed 20260901=${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_sft_v1_seed20260901" \
  --budget 3.0 >>"${LOG}" 2>&1

OUTPUT="${OUTPUT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
paths = (root / "table.json", root / "table.csv", root / "table.md")
for path in paths:
    if not path.is_file():
        raise FileNotFoundError(path)
table = json.loads(paths[0].read_text(encoding="utf-8"))
if table["split"] != "val" or table["test_assets_read"]:
    raise ValueError("Direct-VLM full comparison is not validation-only")
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": "muno21-direct-vlm-full-comparison-complete-v1",
            "status": "complete",
            "files": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in paths
            },
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
