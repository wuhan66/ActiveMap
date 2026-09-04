#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
EPISODES="${STORAGE_ROOT}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
OUTPUT="${STORAGE_ROOT}/processed/muno21_v2/direct_vlm_visuals_val_v1"
LOG="${STORAGE_ROOT}/logs/prepare_muno21_direct_vlm_visuals_val_v1.log"
CONTROL="${STORAGE_ROOT}/processed/muno21_v2/direct_vlm_visuals_val_v1_control"

mkdir -p "${CONTROL}"
exec 9>"${CONTROL}/.lock"
flock -n 9 || exit 0
[[ ! -s "${CONTROL}/COMPLETE.json" ]] || exit 0
rm -f "${CONTROL}/FAILED.json"
trap 'rc=$?; ((rc == 0)) || printf "{\"status\":\"failed\",\"exit_code\":%d,\"split\":\"val\",\"test_assets_read\":false}\n" "${rc}" >"${CONTROL}/FAILED.json"' EXIT

# Avoid competing with active SFT data loading and checkpoint writes.
while pgrep -f "train_agent_sft.py.*muno21_qwen3_4b_balanced_tool" >/dev/null; do
  sleep 60
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" scripts/prepare_muno21_direct_vlm_visuals.py \
  "${EPISODES}" "${OUTPUT}" --split val --image-size 1024 \
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >>"${LOG}" 2>&1

CONTROL="${CONTROL}" OUTPUT="${OUTPUT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

control = Path(os.environ["CONTROL"])
root = Path(os.environ["OUTPUT"])
manifest = root / "manifest.json"
inputs = root / "inputs.jsonl"
labels = root / "labels.jsonl"
data = json.loads(manifest.read_text(encoding="utf-8"))
if data["split"] != "val" or data["test_assets_read"] is not False:
    raise ValueError("full Direct-VLM visual manifest is not validation-only")
if data["sample_count"] != 138:
    raise ValueError(f"unexpected validation support: {data['sample_count']}")
record = {
    "schema_version": "muno21-direct-vlm-visual-preparation-complete-v1",
    "status": "complete",
    "split": "val",
    "sample_count": data["sample_count"],
    "files": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (manifest, inputs, labels)
    },
    "test_assets_read": False,
}
(control / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

trap - EXIT
