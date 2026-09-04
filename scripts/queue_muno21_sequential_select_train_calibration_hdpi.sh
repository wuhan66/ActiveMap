#!/usr/bin/env bash
set -euo pipefail

# Calibrate a fixed Qwen3-VL SELECT adapter using task-disjoint training states
# only, then evaluate that frozen threshold on validation. This is a calibration
# protocol, not a validation threshold sweep and never reads test assets.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
SOURCE="${SOURCE:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/train/selector_balanced_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/val/selector_sft.jsonl}"
ADAPTER="${ADAPTER:-${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_v1_seed20260862/final}"
SPLIT_ROOT="${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/train_calibration_v1"
ROOT="${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_seed20260862_calibrated_v1"
TRAIN_LIKELIHOOD="${ROOT}/train_likelihood"
VAL_LIKELIHOOD="${ROOT}/val_likelihood"
CALIBRATION="${ROOT}/threshold_calibration"
EVALUATION="${ROOT}/validation"
GPU_TRAIN="${GPU_TRAIN:-4}"
GPU_VAL="${GPU_VAL:-5}"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${SOURCE}" "${VAL}" "${ADAPTER}/adapter_model.safetensors"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${ROOT}" ]] || { echo "Existing calibration artifact: ${ROOT}" >&2; exit 3; }
if [[ ! -e "${SPLIT_ROOT}" ]]; then
  PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" \
    scripts/split_sequential_selector_fit_calibration.py "${SOURCE}" "${SPLIT_ROOT}" \
    --calibration-fraction 0.20 --seed 20260866 --positive-multiplier 9 \
    >"${STORE}/logs/muno21_sequential_select_train_calibration_split.log" 2>&1
fi
for path in "${SPLIT_ROOT}/summary.json" "${SPLIT_ROOT}/calibration.jsonl"; do
  [[ -s "${path}" ]] || { echo "Missing split output: ${path}" >&2; exit 4; }
done
mkdir -p "${ROOT}"

run_likelihood() {
  local gpu="$1" source="$2" output="$3" expected="$4" seed="$5" log="$6"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "${PY}" scripts/diagnose_sequential_selector_likelihood.py \
      "${MODEL}" "${ADAPTER}" "${source}" "${output}" --device cuda:0 \
      --expected-records "${expected}" --bootstrap-repetitions 2000 --seed "${seed}" \
      >"${log}" 2>&1
}

# The training-side calibration partition and validation likelihood evaluation
# are disjoint and run concurrently on separate GPUs.
train_rows="$(${PY} - "${SPLIT_ROOT}/calibration.jsonl" <<'PY'
import sys
print(sum(1 for line in open(sys.argv[1], encoding='utf-8') if line.strip()))
PY
)"
run_likelihood "${GPU_TRAIN}" "${SPLIT_ROOT}/calibration.jsonl" "${TRAIN_LIKELIHOOD}" \
  "${train_rows}" 20260866 "${STORE}/logs/muno21_sequential_select_train_likelihood.log" &
train_pid=$!
run_likelihood "${GPU_VAL}" "${VAL}" "${VAL_LIKELIHOOD}" \
  414 20260862 "${STORE}/logs/muno21_sequential_select_val_likelihood.log" &
val_pid=$!
wait "${train_pid}"
wait "${val_pid}"

PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" \
  scripts/calibrate_sequential_selector_threshold.py "${TRAIN_LIKELIHOOD}/traces.jsonl" "${CALIBRATION}" \
  --max-call-rate 0.50 --max-false-call-rate 0.10 --min-recall 0.10 \
  --bootstrap-repetitions 2000 --seed 20260866 \
  >"${STORE}/logs/muno21_sequential_select_threshold_calibration.log" 2>&1
PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" \
  scripts/evaluate_calibrated_sequential_selector.py "${VAL_LIKELIHOOD}/traces.jsonl" \
  "${CALIBRATION}/summary.json" "${EVALUATION}" --expected-records 414 \
  --bootstrap-repetitions 2000 --seed 20260862 \
  >"${STORE}/logs/muno21_sequential_select_calibrated_validation.log" 2>&1

ROOT="${ROOT}" "${PY}" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
outputs = [
    root / "train_likelihood/summary.json",
    root / "val_likelihood/summary.json",
    root / "threshold_calibration/summary.json",
    root / "validation/summary.json",
]
for path in outputs:
    if not path.is_file():
        raise FileNotFoundError(path)
records = [json.loads(path.read_text(encoding="utf-8")) for path in outputs]
if any(record.get("test_assets_read") for record in records):
    raise ValueError("test access detected")
(root / "COMPLETE.json").write_text(json.dumps({
    "schema_version": "sequential-selector-train-calibration-complete-v1",
    "files": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in outputs},
    "test_assets_read": False,
}, indent=2) + "\n", encoding="utf-8")
PY
