#!/usr/bin/env bash
set -euo pipefail

STEP="${1:?usage: run_muno21_qwen3_8b_checkpoint_eval.sh STEP GPU}"
GPU="${2:?usage: run_muno21_qwen3_8b_checkpoint_eval.sh STEP GPU}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="${MUNO21_8B_MODEL:-/home/wh/hf_models/Qwen3-8B}"
RUN="${STORAGE_ROOT}/runs/agent/muno21_qwen3_8b_balanced_tool_sft_seed20260908"
DATA="${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v10_balanced_sparse_tools"
ADAPTER="${RUN}/checkpoints/checkpoint-${STEP}"
OUTPUT="${RUN}/evaluation/capacity_trajectory_v1/checkpoint-${STEP}"

[[ "${STEP}" =~ ^[1-9][0-9]*$ ]] || {
  echo "STEP must be a positive integer" >&2
  exit 2
}
case "${GPU}" in
  1|2|3|4|5|7) ;;
  0|6) echo "GPU${GPU} is reserved" >&2; exit 2 ;;
  *) echo "unsupported physical GPU id: ${GPU}" >&2; exit 2 ;;
esac

while [[ ! -s "${ADAPTER}/adapter_config.json" || ! -s "${ADAPTER}/adapter_model.safetensors" ]]; do
  sleep 30
done

mkdir -p "${OUTPUT}"
exec 9>"${OUTPUT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${OUTPUT}/COMPLETE.json" ]] || exit 0

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_actions.py \
  "${MODEL}" "${DATA}/val/sft_composed.jsonl" "${OUTPUT}/actions" \
  --adapter "${ADAPTER}" --device cuda --batch-size 1 \
  --max-length 2048 --max-new-tokens 64
"${PYTHON}" scripts/analyze_agent_action_errors.py \
  "${OUTPUT}/actions/predictions.jsonl" "${OUTPUT}/actions/error_analysis.json"

OUTPUT="${OUTPUT}" STEP="${STEP}" GPU="${GPU}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
files = {}
for name in ("summary.json", "error_analysis.json"):
    path = root / "actions" / name
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": "muno21-qwen3-8b-capacity-checkpoint-v1",
            "status": "complete",
            "step": int(os.environ["STEP"]),
            "physical_gpu": int(os.environ["GPU"]),
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
