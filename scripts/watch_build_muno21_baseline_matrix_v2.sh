#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
ACQUIRE_ROOT="${STORAGE_ROOT}/runs/agent/muno21_acquire_all_baseline_hdpi_v1"
POLICY_ROOT="${STORAGE_ROOT}/runs/agent/muno21_policy_baselines_hdpi_v2"
HEURISTIC_ROOT="${STORAGE_ROOT}/runs/agent/muno21_heuristic_baselines_hdpi_v1"
TRANSFER_ROOT="${STORAGE_ROOT}/runs/muno21_active_catalog_transfer_v1"
OUTPUT="${STORAGE_ROOT}/paper/muno21_baseline_matrix_hdpi_v3"
STATE="${STORAGE_ROOT}/paper/.muno21_baseline_matrix_hdpi_v3_state"
LOG="${STATE}/watcher.log"

mkdir -p "${STATE}"
exec 9>"${STATE}/.lock"
flock -n 9 || exit 0
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0
exec >>"${LOG}" 2>&1

while [[ ! -s "${ACQUIRE_ROOT}/COMPLETE.json" ]]; do
  if [[ -s "${ACQUIRE_ROOT}/FAILED.json" ]]; then
    cp "${ACQUIRE_ROOT}/FAILED.json" "${OUTPUT}/FAILED.json"
    exit 4
  fi
  sleep 60
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

args=(
  "${OUTPUT}"
  --rollout-summary "${POLICY_ROOT}/rollouts/summary.json"
  --rollout-summary "${HEURISTIC_ROOT}/rollouts/summary.json"
  --rollout-summary "${ACQUIRE_ROOT}/rollouts/summary.json"
  --closed-loop "always_stop=${TRANSFER_ROOT}/closed_loop_always_stop_n414/evaluation/summary.json"
  --closed-loop "qwen_vlm_sft=${TRANSFER_ROOT}/closed_loop_sft_seed20260718_n414/evaluation/summary.json"
  --writeback "oracle=${POLICY_ROOT}/writeback/oracle/summary.json"
  --writeback "generic_selector=${POLICY_ROOT}/writeback/generic_selector/summary.json"
  --writeback "edit_conditioned_selector=${POLICY_ROOT}/writeback/edit_conditioned_selector/summary.json"
  --writeback "random=${HEURISTIC_ROOT}/writeback/random/summary.json"
  --writeback "cheapest=${HEURISTIC_ROOT}/writeback/cheapest/summary.json"
  --writeback "quality_first=${HEURISTIC_ROOT}/writeback/quality_first/summary.json"
  --writeback "uncertainty=${HEURISTIC_ROOT}/writeback/uncertainty/summary.json"
  --writeback "mapex=${HEURISTIC_ROOT}/writeback/mapex/summary.json"
  --writeback "greedy_utility=${HEURISTIC_ROOT}/writeback/greedy_utility/summary.json"
  --writeback "acquire_all=${ACQUIRE_ROOT}/writeback/acquire_all/summary.json"
  --writeback "always_stop=${TRANSFER_ROOT}/writeback_always_stop/evaluation/summary.json"
  --writeback "qwen_vlm_sft=${TRANSFER_ROOT}/writeback_final_sft_seed20260718/evaluation/summary.json"
)
"${PYTHON}" scripts/build_muno21_baseline_table.py "${args[@]}"

OUTPUT="${OUTPUT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
files = {}
for name in ("table.csv", "table.json", "table.md", "table.tex"):
    path = root / name
    if not path.is_file():
        raise FileNotFoundError(path)
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": "muno21-baseline-matrix-v2-complete",
            "status": "complete",
            "split": "val",
            "method_count": 12,
            "files": files,
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
