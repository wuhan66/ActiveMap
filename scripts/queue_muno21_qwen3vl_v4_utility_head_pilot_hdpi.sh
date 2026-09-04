#!/usr/bin/env bash
set -euo pipefail

# Diagnostic only: freeze one formally trained Qwen3-VL adapter, extract its
# observable SELECT-state features, and fit a grouped train-only utility head.
# The validation split is evaluation-only; this script cannot promote C6.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-${STORE}/runs/agent/muno21_qwen3vl_crossfit_supported_weight075_formal_seed20260881/final}"
TRAIN="${TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/train/selector_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/val/selector_sft.jsonl}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_qwen3vl_v4_utility_head_pilot_seed20260881}"
TRAIN_GPU="${TRAIN_GPU:-1}"
VAL_GPU="${VAL_GPU:-2}"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${ADAPTER}/adapter_config.json" "${TRAIN}" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing output: ${OUTPUT}" >&2; exit 3; }
mkdir -p "${STORE}/logs"

extract() {
  local split="$1"
  local input="$2"
  local gpu="$3"
  local output="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    "${PY}" scripts/extract_semantic_vlm_gate_features.py \
      "${MODEL}" "${ADAPTER}" "${input}" "${output}" \
      --stage SELECT --device cuda:0 --batch-size 2 --pooling last_mean \
      >"${STORE}/logs/muno21_qwen3vl_v4_utility_head_${split}.log" 2>&1
}

extract train "${TRAIN}" "${TRAIN_GPU}" "${OUTPUT}/train_features" &
train_pid=$!
extract val "${VAL}" "${VAL_GPU}" "${OUTPUT}/val_features" &
val_pid=$!
wait "${train_pid}"
wait "${val_pid}"

PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" scripts/train_visual_utility_gate.py \
  "${OUTPUT}/train_features" "${OUTPUT}/val_features" "${OUTPUT}/utility_gate" \
  --seed 20260881 --max-call-rate 0.25 --min-oof-recall 0.10 \
  --bootstrap-repetitions 10000 \
  >"${STORE}/logs/muno21_qwen3vl_v4_utility_head_fit.log" 2>&1

OUTPUT="${OUTPUT}" "${PY}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
summary = root / "utility_gate" / "summary.json"
payload = json.loads(summary.read_text(encoding="utf-8"))
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "schema_version": "muno21-qwen3vl-v4-utility-head-pilot-v1",
            "role": "diagnostic-only-train-grouped-structured-action-head",
            "promotion_gate": payload["promotion_gate"],
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
