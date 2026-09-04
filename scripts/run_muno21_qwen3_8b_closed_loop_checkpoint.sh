#!/usr/bin/env bash
set -euo pipefail

LABEL="${1:?usage: run_muno21_qwen3_8b_closed_loop_checkpoint.sh LABEL GPU}"
GPU="${2:?usage: run_muno21_qwen3_8b_closed_loop_checkpoint.sh LABEL GPU}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/agent/muno21_qwen3_8b_balanced_tool_sft_seed20260908"
OUTPUT="${STORAGE_ROOT}/artifacts/paper_rollouts/muno21_qwen3_8b_capacity_closed_loop_v1/${LABEL}"

case "${GPU}" in
  1|2|3|4|5|7) ;;
  0|6) echo "GPU${GPU} is reserved" >&2; exit 2 ;;
  *) echo "unsupported physical GPU id: ${GPU}" >&2; exit 2 ;;
esac
if [[ "${LABEL}" == "final" ]]; then
  ADAPTER="${RUN}/final"
elif [[ "${LABEL}" =~ ^checkpoint-([1-9][0-9]*)$ ]]; then
  ADAPTER="${RUN}/checkpoints/${LABEL}"
else
  echo "LABEL must be final or checkpoint-N" >&2
  exit 2
fi
for path in "${ADAPTER}/adapter_config.json" "${ADAPTER}/adapter_model.safetensors"; do
  [[ -s "${path}" ]] || { echo "missing adapter input: ${path}" >&2; exit 3; }
done
[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing to overwrite ${OUTPUT}" >&2
  exit 3
}

cd "${PROJECT_ROOT}"
MUNO21_AGENT_MODEL="/home/wh/hf_models/Qwen3-8B" \
MUNO21_AGENT_SEED=20260908 \
MUNO21_SELECTOR_SEED=20260811 \
MUNO21_V12_GPU="${GPU}" \
MUNO21_V12_ADAPTER="${ADAPTER}" \
MUNO21_V12_ROLLOUT_ROOT="${OUTPUT}" \
MUNO21_V12_METHODS="qwen3_4b_sft_calibrated_tool_to_belief,edit_conditioned_proactive_tools" \
MUNO21_V12_TOOL_NEED_THRESHOLD=0.09 \
MUNO21_V12_ASSESS=0 \
  bash scripts/run_muno21_v12_proactive_rollout.sh

"${PYTHON}" scripts/compare_agent_rollouts.py \
  "${OUTPUT}/edit_conditioned_proactive_tools.jsonl" \
  "${OUTPUT}/qwen3_4b_sft_calibrated_tool_to_belief.jsonl" \
  "${OUTPUT}/paired_vs_edit.json" \
  --bootstrap 5000 --seed 20260913

OUTPUT="${OUTPUT}" LABEL="${LABEL}" GPU="${GPU}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
files = {}
for name in ("summary.json", "paired_vs_edit.json"):
    path = root / name
    files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "schema_version": "muno21-qwen3-8b-capacity-closed-loop-v1",
            "status": "complete",
            "label": os.environ["LABEL"],
            "physical_gpu": int(os.environ["GPU"]),
            "model": "/home/wh/hf_models/Qwen3-8B",
            "tool_need_threshold": 0.09,
            "bootstrap_repetitions": 5000,
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
