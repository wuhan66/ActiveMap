#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
GPU="${GPU:-3}"
SEED="${SEED:-20260831}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
RUN_FAMILY="${RUN_FAMILY:-muno21_direct_vlm_qwen3vl4b_sft_v1}"
EPISODES="${STORAGE_ROOT}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl"
DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/direct_vlm_sft_v1"
RUN_ROOT="${STORAGE_ROOT}/runs/agent/${RUN_FAMILY}_seed${SEED}"
UPDATER="${STORAGE_ROOT}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt"
LOG="${RUN_ROOT}/run.log"

mkdir -p "${RUN_ROOT}"
exec 9>"${RUN_ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -s "${RUN_ROOT}/COMPLETE.json" ]] || exit 0

fail() {
  status=$?
  if [[ "${status}" -ne 0 ]]; then
    printf '{"status":"failed","exit_code":%d}\n' "${status}" >"${RUN_ROOT}/FAILED.json"
  fi
}
trap fail EXIT

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

for split in train val; do
  visual_root="${DATA_ROOT}/visuals_${split}"
  if [[ ! -s "${visual_root}/manifest.json" ]]; then
    "${PYTHON}" scripts/prepare_muno21_direct_vlm_visuals.py \
      "${EPISODES}" "${visual_root}" --split "${split}" --image-size 512 \
      --composite-panel-size 384 \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >>"${LOG}" 2>&1
  fi
  dataset="${DATA_ROOT}/${split}.jsonl"
  if [[ ! -s "${dataset}" ]]; then
    build_args=()
    [[ "${split}" != "train" ]] || build_args=(--balance)
    "${PYTHON}" scripts/build_muno21_direct_vlm_sft.py \
      "${visual_root}/inputs.jsonl" "${visual_root}/labels.jsonl" \
      "${dataset}" --split "${split}" "${build_args[@]}" >>"${LOG}" 2>&1
  fi
done

if [[ ! -s "${RUN_ROOT}/preflight/data_summary.json" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/train_semantic_vlm_sft.py \
    "${MODEL}" "${DATA_ROOT}/train.jsonl" "${RUN_ROOT}/preflight" \
    --eval-jsonl "${DATA_ROOT}/val.jsonl" --max-length 1536 \
    --max-train-samples 2 --max-eval-samples 2 --dry-run >>"${LOG}" 2>&1
fi

while [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid \
  --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  sleep 30
done

export CUDA_VISIBLE_DEVICES="${GPU}"
"${PYTHON}" scripts/train_semantic_vlm_sft.py \
  "${MODEL}" "${DATA_ROOT}/train.jsonl" "${RUN_ROOT}/training" \
  --eval-jsonl "${DATA_ROOT}/val.jsonl" --epochs 3 \
  --learning-rate 1e-4 --batch-size 1 --gradient-accumulation 16 \
  --max-length 1536 --lora-rank 16 --lora-alpha 32 --seed "${SEED}" \
  --logging-steps 5 --eval-steps 50 --save-steps 50 --save-total-limit 3 \
  --early-stopping-patience 3 --teacher-loss-model-selection \
  --dataloader-num-workers 2 --dataloader-prefetch-factor 2 \
  --dataloader-persistent-workers >>"${LOG}" 2>&1

"${PYTHON}" scripts/evaluate_muno21_direct_vlm.py \
  "${MODEL}" "${DATA_ROOT}/visuals_val/inputs.jsonl" \
  "${DATA_ROOT}/visuals_val/labels.jsonl" "${RUN_ROOT}/evaluation" \
  --adapter "${RUN_ROOT}/training/final" --device cuda:0 \
  --image-mode composite --prompt-version operational_v2 >>"${LOG}" 2>&1

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
    root / "training/data_summary.json",
    root / "training/train_metrics.json",
    root / "training/eval_metrics.json",
    root / "evaluation/summary.json",
    root / "evaluation/traces.jsonl",
    root / "writeback/default/summary.json",
    root / "writeback/safe_delta/summary.json",
)
for path in paths:
    if not path.is_file():
        raise FileNotFoundError(path)
summary = json.loads((root / "evaluation/summary.json").read_text(encoding="utf-8"))
protocol = summary["protocol"]
if protocol["split"] != "val" or protocol["image_mode"] != "composite":
    raise ValueError("unexpected supervised Direct-VLM evaluation protocol")
if summary["adapter"] is None or summary["test_assets_read"]:
    raise ValueError("adapter provenance or leakage audit failed")
record = {
    "schema_version": "muno21-direct-vlm-supervised-sft-complete-v1",
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

rm -f "${RUN_ROOT}/FAILED.json"
trap - EXIT
