#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:?SEED is required}"
GPU="${GPU:?GPU is required}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
THRESHOLD_TAG="${MUNO21_V12_THRESHOLD_TAG:-threshold009}"
ROLLOUT_ROOT="${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v12_proactive_tool_four_seed_${THRESHOLD_TAG}_v1/seed${SEED}"
OUTPUT_ROOT="${STORAGE_ROOT}/runs/agent/muno21_v12_${THRESHOLD_TAG}_writeback_v1/seed${SEED}"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
EPISODES="${STORAGE_ROOT}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"

case "${GPU}" in
  1|2|3|4|5|7) ;;
  0|6) echo "GPU${GPU} is reserved" >&2; exit 2 ;;
  *) echo "unsupported physical GPU id: ${GPU}" >&2; exit 2 ;;
esac
for path in "${UPDATER}" "${EPISODES}" "${ROLLOUT_ROOT}/summary.json"; do
  [[ -s "${path}" ]] || { echo "missing input: ${path}" >&2; exit 3; }
done

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${OUTPUT_ROOT}/COMPLETE.json" ]] || exit 0
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

methods=(
  qwen3_4b_sft_calibrated_tool_to_belief
  edit_conditioned_proactive_tools
)
for method in "${methods[@]}"; do
  rollout="${ROLLOUT_ROOT}/${method}.jsonl"
  [[ -s "${rollout}" ]] || { echo "missing rollout: ${rollout}" >&2; exit 3; }
  raw="${OUTPUT_ROOT}/${method}_raw"
  if [[ ! -s "${raw}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
      "${UPDATER}" "${EPISODES}" "${rollout}" "${raw}" \
      --device cuda:0 --split val --image-size 512 --threshold 0.5 \
      --protocol-name "muno21-v12-${THRESHOLD_TAG}-${method}-raw-v1" \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  fi
  safe="${OUTPUT_ROOT}/${method}_safe_delta"
  if [[ ! -s "${safe}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_map_writeback.py \
      "${UPDATER}" "${EPISODES}" "${rollout}" "${safe}" \
      --device cuda:0 --split val --image-size 512 --threshold 0.5 \
      --add-min-delta-component-pixels 1536 \
      --delete-min-delta-component-pixels 1024 \
      --reshape-min-delta-component-pixels 0 \
      --preserve-largest-delta-component \
      --protocol-name "muno21-v12-${THRESHOLD_TAG}-${method}-safe-delta-v1" \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  fi
done

SEED="${SEED}" GPU="${GPU}" OUTPUT_ROOT="${OUTPUT_ROOT}" THRESHOLD_TAG="${THRESHOLD_TAG}" "${PYTHON}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
methods = (
    "qwen3_4b_sft_calibrated_tool_to_belief",
    "edit_conditioned_proactive_tools",
)
files = {}
for method in methods:
    for variant in ("raw", "safe_delta"):
        path = root / f"{method}_{variant}" / "summary.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        files[f"{method}_{variant}"] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
record = {
    "schema_version": f"muno21-v12-{os.environ['THRESHOLD_TAG']}-writeback-seed-v1",
    "status": "complete",
    "seed": int(os.environ["SEED"]),
    "physical_gpu": int(os.environ["GPU"]),
    "files": files,
    "split": "val",
    "test_assets_read": False,
}
(root / "COMPLETE.json").write_text(
    json.dumps(record, indent=2) + "\n", encoding="utf-8"
)
PY
