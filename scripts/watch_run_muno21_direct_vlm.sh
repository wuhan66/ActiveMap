#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${GPU:-3}"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
VISUAL_ROOT="${STORAGE_ROOT}/processed/muno21_v2/direct_vlm_visuals_val_v1"
VISUAL_CONTROL="${STORAGE_ROOT}/processed/muno21_v2/direct_vlm_visuals_val_v1_control"
ACQUIRE_CONTROL="${STORAGE_ROOT}/runs/agent/muno21_acquire_all_baseline_hdpi_v1"
RUN_ROOT="${STORAGE_ROOT}/runs/agent/muno21_direct_vlm_qwen3vl4b_zeroshot_v1"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
EPISODES="${STORAGE_ROOT}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
LOG="${RUN_ROOT}/run.log"

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0
rm -f "${RUN_ROOT}/FAILED.json"
trap 'rc=$?; ((rc == 0)) || printf "{\"status\":\"failed\",\"exit_code\":%d,\"split\":\"val\",\"test_assets_read\":false}\n" "${rc}" >"${RUN_ROOT}/FAILED.json"' EXIT

wait_for_complete() {
  local root="$1"
  while [[ ! -s "${root}/COMPLETE.json" ]]; do
    [[ ! -s "${root}/FAILED.json" ]] || {
      echo "upstream failed: ${root}" >&2
      exit 4
    }
    sleep 60
  done
}
wait_for_complete "${VISUAL_CONTROL}"
wait_for_complete "${ACQUIRE_CONTROL}"

# Claim GPU3 only after it remains free and the project uses at most four
# physical cards. This serializes behind seed24 and acquire-all writeback.
idle_polls=0
while ((idle_polls < 6)); do
  mapfile -t memory < <(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
  )
  active=0
  for used in "${memory[@]}"; do ((used >= 512)) && active=$((active + 1)); done
  if ((memory[GPU] < 512 && active <= 4)); then
    idle_polls=$((idle_polls + 1))
  else
    idle_polls=0
  fi
  sleep 10
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${GPU}"

if [[ ! -s "${RUN_ROOT}/smoke2/summary.json" ]]; then
  "${PYTHON}" scripts/evaluate_muno21_direct_vlm.py \
    "${MODEL}" "${VISUAL_ROOT}/inputs.jsonl" "${VISUAL_ROOT}/labels.jsonl" \
    "${RUN_ROOT}/smoke2" --device cuda:0 --limit 2 >>"${LOG}" 2>&1
fi

RUN_ROOT="${RUN_ROOT}" "${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

summary = json.loads(
    (Path(os.environ["RUN_ROOT"]) / "smoke2/summary.json").read_text(encoding="utf-8")
)
if summary["sample_count"] != 2:
    raise ValueError("Direct VLM smoke has unexpected support")
if summary["schema_valid_rate"] < 0.5:
    raise ValueError("Direct VLM smoke failed structured-action viability")
if summary["test_assets_read"] is not False:
    raise ValueError("Direct VLM smoke is not validation-only")
PY

if [[ ! -s "${RUN_ROOT}/evaluation/summary.json" ]]; then
  "${PYTHON}" scripts/evaluate_muno21_direct_vlm.py \
    "${MODEL}" "${VISUAL_ROOT}/inputs.jsonl" "${VISUAL_ROOT}/labels.jsonl" \
    "${RUN_ROOT}/evaluation" --device cuda:0 >>"${LOG}" 2>&1
fi

if [[ ! -s "${RUN_ROOT}/writeback/default/summary.json" ]]; then
  "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${RUN_ROOT}/evaluation/rollouts.jsonl" \
    "${RUN_ROOT}/writeback/default" --device cuda:0 --split val \
    --image-size 512 --threshold 0.5 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >>"${LOG}" 2>&1
fi

if [[ ! -s "${RUN_ROOT}/writeback/safe_delta/summary.json" ]]; then
  "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${RUN_ROOT}/evaluation/rollouts.jsonl" \
    "${RUN_ROOT}/writeback/safe_delta" --device cuda:0 --split val \
    --image-size 512 --threshold 0.5 \
    --add-min-delta-component-pixels 1536 \
    --delete-min-delta-component-pixels 1024 \
    --reshape-min-delta-component-pixels 0 \
    --preserve-largest-delta-component \
    --protocol-name "muno21-operation-conditioned-safe-delta-v1" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >>"${LOG}" 2>&1
fi

RUN_ROOT="${RUN_ROOT}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["RUN_ROOT"])
paths = {
    "evaluation": root / "evaluation/summary.json",
    "rollouts": root / "evaluation/rollouts.jsonl",
    "default_writeback": root / "writeback/default/summary.json",
    "safe_delta_writeback": root / "writeback/safe_delta/summary.json",
}
for path in paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)
record = {
    "schema_version": "muno21-direct-vlm-complete-v1",
    "status": "complete",
    "split": "val",
    "method": "direct_vlm",
    "files": {
        name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for name, path in paths.items()
    },
    "test_assets_read": False,
}
(root / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY

trap - EXIT
