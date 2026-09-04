#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
NATIVE="${STORE}/runs/argotweak/native_adapter_frozen_baseline_v2/episodes"
VISUAL="${STORE}/runs/argotweak/qwen3vl_frozen_features_v1"
ROOT="${STORE}/runs/argotweak/map_conditioned_selector_screen_v1"
mkdir -p "${ROOT}/logs" "${ROOT}/status"
exec 9>"${ROOT}/.lock"; flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

for split in train val; do
  "${PYTHON}" scripts/build_argotweak_observable_features.py \
    --episodes "${NATIVE}/${split}.jsonl" \
    --output-dir "${ROOT}/features/structured/${split}" \
    >"${ROOT}/logs/build_structured_${split}.log" 2>&1
  "${PYTHON}" scripts/build_argotweak_observable_features.py \
    --episodes "${NATIVE}/${split}.jsonl" --visual-features "${VISUAL}/${split}" \
    --output-dir "${ROOT}/features/qwen_structured/${split}" \
    >"${ROOT}/logs/build_qwen_structured_${split}.log" 2>&1
done

run_variant() {
  local gpu="$1" variant="$2"
  local feature_root="${ROOT}/features/${variant}"
  local selector="${ROOT}/selector/${variant}_seed20261511"
  local validation="${ROOT}/validation/${variant}_seed20261511"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/train_argotweak_visual_selector.py \
    --features "${feature_root}/train" --output-dir "${selector}" \
    --seed 20261511 --device cuda >"${ROOT}/logs/train_${variant}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
    --features "${feature_root}/val" --checkpoint "${selector}/best.pt" \
    --output-dir "${validation}" --device cuda >"${ROOT}/logs/val_${variant}.log" 2>&1
}

run_variant 0 structured & pid0=$!
run_variant 1 qwen_structured & pid1=$!
status=0
wait "${pid0}" || status=1
wait "${pid1}" || status=1
if [[ "${status}" -ne 0 ]]; then touch "${ROOT}/QUEUE_FAILED"; exit 1; fi
touch "${ROOT}/QUEUE_COMPLETED"
