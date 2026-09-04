#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="${MUNO21_8B_MODEL:-/home/wh/hf_models/Qwen3-8B}"
RUN="${STORAGE_ROOT}/runs/agent/muno21_qwen3_8b_balanced_tool_sft_seed20260908"
DATA="${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools"
OUTPUT="${RUN}/evaluation/capacity_static_v1"
GPU="${GPU:-5}"
LOG="${OUTPUT}/watcher.log"
POLL_SECONDS="${POLL_SECONDS:-60}"

case "${GPU}" in
  1|2|3|4|5|7) ;;
  0|6) echo "GPU${GPU} is reserved" >&2; exit 2 ;;
  *) echo "unsupported physical GPU id: ${GPU}" >&2; exit 2 ;;
esac
mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0
exec >>"${LOG}" 2>&1

while [[ ! -s "${RUN}/final/adapter_config.json" ]]; do
  if [[ -s "${RUN}/run_state.json" ]]; then
    status="$("${PYTHON}" -c \
      'import json,sys; print(json.load(open(sys.argv[1])).get("status","unknown"))' \
      "${RUN}/run_state.json")"
    [[ "${status}" != "failed" ]] || {
      printf '{"status":"failed","reason":"training_failed","test_assets_read":false}\n' \
        >"${OUTPUT}/FAILED.json"
      exit 4
    }
  fi
  sleep "${POLL_SECONDS}"
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
actions="${OUTPUT}/actions"
if [[ ! -s "${actions}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_actions.py \
    "${MODEL}" "${DATA}/val/sft_composed.jsonl" "${actions}" \
    --adapter "${RUN}/final" --device cuda --batch-size 1 \
    --max-length 2048 --max-new-tokens 64
fi
if [[ ! -s "${actions}/error_analysis.json" ]]; then
  "${PYTHON}" scripts/analyze_agent_action_errors.py \
    "${actions}/predictions.jsonl" "${actions}/error_analysis.json"
fi

OUTPUT="${OUTPUT}" RUN="${RUN}" GPU="${GPU}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
summary = root / "actions" / "summary.json"
errors = root / "actions" / "error_analysis.json"
record = {
    "schema_version": "muno21-qwen3-8b-capacity-static-v1",
    "status": "complete",
    "physical_gpu": int(os.environ["GPU"]),
    "run": os.environ["RUN"],
    "files": {
        "summary.json": hashlib.sha256(summary.read_bytes()).hexdigest(),
        "error_analysis.json": hashlib.sha256(errors.read_bytes()).hexdigest(),
    },
    "split": "val",
    "test_assets_read": False,
}
(root / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY
