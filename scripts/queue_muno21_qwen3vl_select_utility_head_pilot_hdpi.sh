#!/usr/bin/env bash
set -euo pipefail

# One causal A1 repair: use a train-only calibrated utility head over frozen
# Qwen3-VL SELECT states, rather than free-form generated STOP/ACQUIRE tokens.
# This is a diagnostic/promotion screen only. It neither reads test nor starts
# joint SFT, recurrent rollouts, or GRPO.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_v1_seed20260862/final}"
TRAIN="${TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/train/selector_balanced_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/val/selector_sft.jsonl}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_qwen3vl4b_select_utility_head_pilot_seed20260862}"
TRAIN_GPU="${TRAIN_GPU:-1}"
VAL_GPU="${VAL_GPU:-2}"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${ADAPTER}/adapter_config.json" "${TRAIN}" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing output: ${OUTPUT}" >&2; exit 3; }
mkdir -p "${STORE}/logs"

extract() {
  local split="$1" input="$2" gpu="$3" output="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    "${PY}" scripts/extract_semantic_vlm_gate_features.py \
      "${MODEL}" "${ADAPTER}" "${input}" "${output}" \
      --stage SELECT --device cuda:0 --batch-size 2 --pooling last_mean \
      >"${STORE}/logs/muno21_qwen3vl_select_utility_head_${split}.log" 2>&1
}

extract train "${TRAIN}" "${TRAIN_GPU}" "${OUTPUT}/train_features" &
train_pid=$!
extract val "${VAL}" "${VAL_GPU}" "${OUTPUT}/val_features" &
val_pid=$!
wait "${train_pid}"
wait "${val_pid}"

PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" scripts/train_visual_utility_gate.py \
  "${OUTPUT}/train_features" "${OUTPUT}/val_features" "${OUTPUT}/utility_gate" \
  --seed 20260867 --max-call-rate 0.25 --min-oof-recall 0.10 \
  --bootstrap-repetitions 10000 \
  >"${STORE}/logs/muno21_qwen3vl_select_utility_head_fit.log" 2>&1

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
            "schema_version": "muno21-qwen3vl-select-utility-head-pilot-v1",
            "role": "A1 single-seed structured-action-head diagnostic",
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
