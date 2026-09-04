#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${GPU:-3}"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
VISUAL_ROOT="${STORAGE_ROOT}/processed/muno21_v2/direct_vlm_visuals_val_v1"
UPSTREAM="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_zeroshot_v1"
RUN_ROOT="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_operational_prompt_v2"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
EPISODES="${STORAGE_ROOT}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
LOG="${RUN_ROOT}/run.log"

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0

while [[ ! -s "${UPSTREAM}/COMPLETE.json" ]]; do
  [[ ! -s "${UPSTREAM}/FAILED.json" ]] || exit 4
  sleep 30
done
while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

"${PYTHON}" scripts/evaluate_muno21_direct_vlm.py \
  "${MODEL}" "${VISUAL_ROOT}/inputs.jsonl" "${VISUAL_ROOT}/labels.jsonl" \
  "${RUN_ROOT}/evaluation" --device cuda:0 \
  --prompt-version operational_v2 >>"${LOG}" 2>&1

for mode in default safe_delta; do
  args=()
  if [[ "${mode}" == "safe_delta" ]]; then
    args=(
      --add-min-delta-component-pixels 1536
      --delete-min-delta-component-pixels 1024
      --reshape-min-delta-component-pixels 0
      --preserve-largest-delta-component
      --protocol-name muno21-operation-conditioned-safe-delta-v1
    )
  fi
  "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${RUN_ROOT}/evaluation/rollouts.jsonl" \
    "${RUN_ROOT}/writeback/${mode}" --device cuda:0 --split val \
    --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    "${args[@]}" >>"${LOG}" 2>&1
done

RUN_ROOT="${RUN_ROOT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
paths = (
    root / "evaluation/summary.json",
    root / "evaluation/traces.jsonl",
    root / "writeback/default/summary.json",
    root / "writeback/safe_delta/summary.json",
)
for path in paths:
    if not path.is_file():
        raise FileNotFoundError(path)
summary = json.loads(paths[0].read_text(encoding="utf-8"))
if summary["protocol"]["prompt_version"] != "operational_v2":
    raise ValueError("unexpected Direct-VLM prompt protocol")
record = {
    "schema_version": "muno21-direct-vlm-operational-prompt-v2-complete-v1",
    "status": "complete",
    "split": "val",
    "files": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    },
    "test_assets_read": False,
}
(root / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY
