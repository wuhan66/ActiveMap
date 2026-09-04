#!/usr/bin/env bash
set -euo pipefail

# One pre-registered validation-only smoke. It must not be expanded to a seed
# matrix unless its frozen validation result is safe, positive, and task-CI
# supported. The encoder features are frozen from the formal v4 seed 20260881.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
GPU="${GPU:-1}"
RUN_NAME="${RUN_NAME:-muno21_vla_structured_set_action_head_smoke_seed20260891}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/${RUN_NAME}}"
TRAIN_FEATURES="${TRAIN_FEATURES:-${STORE}/runs/agent/muno21_qwen3vl_v4_utility_head_pilot_seed20260881/train_features}"
VAL_FEATURES="${VAL_FEATURES:-${STORE}/runs/agent/muno21_qwen3vl_v4_utility_head_pilot_seed20260881/val_features}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/train/selector_sft.jsonl}"
VAL_MANIFEST="${VAL_MANIFEST:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/val/selector_sft.jsonl}"
TRAIN_AUDIT="${TRAIN_AUDIT:-${STORE}/runs/agent/muno21_qwen3vl_v4_public_context_audit_train_20260806_v2}"
VAL_AUDIT="${VAL_AUDIT:-${STORE}/runs/agent/muno21_qwen3vl_v4_public_context_audit_20260806_v2}"
LOG="${LOG:-${STORE}/logs/${RUN_NAME}.log}"
FEATURE_MODE="${FEATURE_MODE:-full}"
DISABLE_SET_CONTEXT="${DISABLE_SET_CONTEXT:-0}"
MAXIMUM_FALSE_CALL_RATE="${MAXIMUM_FALSE_CALL_RATE:-0.10}"
SEED="${SEED:-20260891}"

cd "${PROJECT}"
for path in "${PY}" "${TRAIN_FEATURES}/features.npy" "${VAL_FEATURES}/features.npy" "${TRAIN_MANIFEST}" "${VAL_MANIFEST}" "${TRAIN_AUDIT}/summary.json" "${VAL_AUDIT}/summary.json"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing output: ${OUTPUT}" >&2; exit 3; }
mkdir -p "${STORE}/logs"

command=(
  "${PY}" scripts/train_vla_structured_set_action_head.py
  "${TRAIN_FEATURES}" "${VAL_FEATURES}" "${TRAIN_MANIFEST}" "${VAL_MANIFEST}" \
  "${TRAIN_AUDIT}" "${VAL_AUDIT}" "${OUTPUT}" \
  --device cuda:0 --seed "${SEED}" --epochs 80 --batch-size 32 --learning-rate 0.0003 \
  --projection-dim 64 --hidden-dim 128 --dropout 0.15 --folds 5 \
  --positive-sampling-fraction 0.25 --positive-weight 4.0 --temperature 0.25 \
  --regression-weight 1.0 --listwise-weight 1.0 --gate-weight 1.0 \
  --maximum-false-call-rate "${MAXIMUM_FALSE_CALL_RATE}" --bootstrap-repetitions 10000 \
  --feature-mode "${FEATURE_MODE}"
)
if [[ "${DISABLE_SET_CONTEXT}" == "1" ]]; then
  command+=(--disable-set-context)
fi
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  "${command[@]}" >"${LOG}" 2>&1

OUTPUT="${OUTPUT}" "${PY}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT"])
summary = root / "summary.json"
payload = json.loads(summary.read_text(encoding="utf-8"))
(root / "COMPLETE.json").write_text(
    json.dumps(
        {
            "status": "complete",
            "role": "diagnostic-only-structured-vla-set-action-head-smoke",
            "promotion_gate": payload["promotion_gate"],
            "summary_sha256": hashlib.sha256(summary.read_bytes()).hexdigest(),
            "test_assets_read": False,
        },
        indent=2,
    ) + "\n",
    encoding="utf-8",
)
PY
