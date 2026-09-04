#!/usr/bin/env bash
set -euo pipefail

# Validation-only Qwen3-VL feature screen for the ArgoTweak portability pilot.
# GPU4 is reserved for this queue while the SN7 visual-ranker screen uses 1-3.
PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
NATIVE="${STORE}/runs/argotweak/native_adapter_frozen_baseline_v2/episodes"
FEATURE_ROOT="${STORE}/runs/argotweak/qwen3vl_frozen_features_screen_20260805"
SELECTOR="${STORE}/runs/argotweak/qwen3vl_visual_selector_screen_20260805"
ROOT="${STORE}/runs/queues/argotweak_qwen_vla_screen_20260805"
GPU="${GPU:-4}"

mkdir -p "${ROOT}/logs" "${ROOT}/status" "${FEATURE_ROOT}"
exec 9>"${ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0
[[ ! -e "${ROOT}/QUEUE_FAILED" ]] || exit 0

cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU}"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

for path in \
  "${PYTHON}" "${MODEL}/config.json" \
  "${NATIVE}/train.jsonl" "${NATIVE}/val.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing input: ${path}" >&2; touch "${ROOT}/QUEUE_FAILED"; exit 3; }
done

run_features() {
  local split="$1"
  local output="${FEATURE_ROOT}/${split}"
  if [[ ! -s "${output}/summary.json" ]]; then
    "${PYTHON}" -u scripts/extract_argotweak_qwen3vl_features.py \
      --episodes "${NATIVE}/${split}.jsonl" \
      --model "${MODEL}" \
      --output-dir "${output}" \
      --device cuda --batch-size 2 --num-shards 1 --shard-index 0 \
      >"${ROOT}/logs/features_${split}.log" 2>&1
  fi
  touch "${ROOT}/status/features_${split}.done"
}

run_features train
run_features val

if [[ ! -s "${SELECTOR}/summary.json" ]]; then
  "${PYTHON}" scripts/train_argotweak_visual_selector.py \
    --features "${FEATURE_ROOT}/train" \
    --output-dir "${SELECTOR}" \
    --seed 20261550 --device cuda \
    >"${ROOT}/logs/selector_train.log" 2>&1
fi

VALIDATION="${SELECTOR}/validation"
if [[ ! -s "${VALIDATION}/summary.json" ]]; then
  "${PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
    --features "${FEATURE_ROOT}/val" \
    --checkpoint "${SELECTOR}/best.pt" \
    --output-dir "${VALIDATION}" --device cuda \
    >"${ROOT}/logs/selector_validation.log" 2>&1
fi

EXECUTABLE="${SELECTOR}/executable"
if [[ ! -s "${EXECUTABLE}/summary.json" ]]; then
  "${PYTHON}" scripts/evaluate_argotweak_executable_policies.py \
    --episodes "${NATIVE}/val.jsonl" \
    --rankings "${VALIDATION}/rankings.jsonl" \
    --budgets 1,3,5,10 --commit-confidence 0.50 \
    --output-dir "${EXECUTABLE}" \
    >"${ROOT}/logs/executable.log" 2>&1
fi

"${GIS_PYTHON}" - <<PY >"${ROOT}/logs/promotion.json" 2>&1
import json
from pathlib import Path

root = Path("${ROOT}")
summary = json.loads((Path("${EXECUTABLE}") / "summary.json").read_text())
top5 = next(row for row in summary if row.get("policy") == "learned_top5")
promoted = bool(
    top5.get("balanced_utility", -1.0) > 0.0
    and top5.get("atomic_edit_precision", 0.0) >= 0.70
)
(root / "status" / ("PROMOTED" if promoted else "NOT_PROMOTED")).touch()
print(json.dumps({"promoted": promoted, "policy": top5}, indent=2))
PY

printf '%s\n' '{"schema_version":"argotweak-qwen-vla-screen-v1","split":"val","test_assets_read":false,"seed":20261550,"gpu":'"${GPU}"'}' \
  >"${SELECTOR}/protocol.json"
date -Is >"${ROOT}/QUEUE_COMPLETED"
