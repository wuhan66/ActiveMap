#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
THRESHOLD_TAG="${MUNO21_V12_THRESHOLD_TAG:-threshold009}"
RUN="${STORAGE_ROOT}/runs/agent/muno21_v12_${THRESHOLD_TAG}_writeback_v1"
OUTPUT="${STORAGE_ROOT}/artifacts/paper_evidence/muno21_v12_${THRESHOLD_TAG}_writeback_4seed_v1"
SEEDS=(20260822 20260823 20260824 20260825)

mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

qwen_raw_vs_edit=()
qwen_safe_vs_edit=()
qwen_safe_vs_raw=()
for seed in "${SEEDS[@]}"; do
  root="${RUN}/seed${seed}"
  qwen_raw="${root}/qwen3_4b_sft_calibrated_tool_to_belief_raw/writeback.jsonl"
  qwen_safe="${root}/qwen3_4b_sft_calibrated_tool_to_belief_safe_delta/writeback.jsonl"
  edit_raw="${root}/edit_conditioned_proactive_tools_raw/writeback.jsonl"
  edit_safe="${root}/edit_conditioned_proactive_tools_safe_delta/writeback.jsonl"
  for path in "${qwen_raw}" "${qwen_safe}" "${edit_raw}" "${edit_safe}"; do
    [[ -s "${path}" ]] || { echo "missing writeback: ${path}" >&2; exit 3; }
  done
  qwen_raw_vs_edit+=(--pair "${seed}=${edit_raw},${qwen_raw}")
  qwen_safe_vs_edit+=(--pair "${seed}=${edit_safe},${qwen_safe}")
  qwen_safe_vs_raw+=(--pair "${seed}=${qwen_raw},${qwen_safe}")
done

"${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
  "${OUTPUT}/qwen_raw_vs_edit_raw.json" "${qwen_raw_vs_edit[@]}" \
  --repetitions 10000 --seed 20260908
"${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
  "${OUTPUT}/qwen_safe_vs_edit_safe.json" "${qwen_safe_vs_edit[@]}" \
  --repetitions 10000 --seed 20260909
"${PYTHON}" scripts/aggregate_agent_writeback_pairs.py \
  "${OUTPUT}/qwen_safe_vs_qwen_raw.json" "${qwen_safe_vs_raw[@]}" \
  --repetitions 10000 --seed 20260910

OUTPUT="${OUTPUT}" THRESHOLD_TAG="${THRESHOLD_TAG}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
names = (
    "qwen_raw_vs_edit_raw.json",
    "qwen_safe_vs_edit_safe.json",
    "qwen_safe_vs_qwen_raw.json",
)
files = {}
for name in names:
    path = root / name
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": f"muno21-v12-{os.environ['THRESHOLD_TAG']}-writeback-4seed-v1",
            "status": "complete",
            "model_seeds": [20260822, 20260823, 20260824, 20260825],
            "bootstrap_repetitions": 10000,
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
