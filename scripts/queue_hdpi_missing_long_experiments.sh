#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
GIS_PYTHON="${STORE}/envs/activemap-gis/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
NATIVE="${STORE}/runs/argotweak/native_adapter_frozen_baseline_v2/episodes"
ROOT="${STORE}/runs/queues/hdpi_missing_long_20260804"
FEATURE_ROOT="${STORE}/runs/argotweak/qwen3vl_frozen_features_v1"
mkdir -p "${ROOT}/logs" "${ROOT}/status" "${FEATURE_ROOT}"
exec 9>"${ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"

run_feature_shard() {
  local gpu="$1" shard="$2"
  for split in train val; do
    local output="${FEATURE_ROOT}/shards/${split}_${shard}"
    if [[ ! -s "${output}/summary.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/extract_argotweak_qwen3vl_features.py \
        --episodes "${NATIVE}/${split}.jsonl" --model "${MODEL}" --output-dir "${output}" \
        --device cuda --batch-size 4 --num-shards 4 --shard-index "${shard}" \
        >"${ROOT}/logs/features_${split}_${shard}.log" 2>&1
    fi
  done
  touch "${ROOT}/status/features_${shard}.done"
}

# Five simultaneous GPU jobs: one native evaluator and four frozen feature shards.
GPU=0 bash scripts/launch_argotweak_seed2_eval_hdpi.sh \
  >"${ROOT}/logs/argotweak_seed2_eval.log" 2>&1 & pid_eval=$!
pids=()
for shard in 0 1 2 3; do
  gpu=$((shard + 1))
  run_feature_shard "${gpu}" "${shard}" &
  pids+=("$!")
  sleep 8
done
status=0
wait "${pid_eval}" || status=1
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
if [[ "${status}" -ne 0 ]]; then touch "${ROOT}/QUEUE_FAILED"; exit 1; fi
touch "${ROOT}/status/parallel_long_stage.done"

for split in train val; do
  merged="${FEATURE_ROOT}/${split}"
  if [[ ! -s "${merged}/summary.json" ]]; then
    inputs=()
    for shard in 0 1 2 3; do inputs+=("${FEATURE_ROOT}/shards/${split}_${shard}"); done
    "${PYTHON}" scripts/merge_argotweak_feature_shards.py \
      --inputs "${inputs[@]}" --output-dir "${merged}" \
      >"${ROOT}/logs/merge_${split}.log" 2>&1
  fi
done
touch "${ROOT}/status/features_merged.done"

SELECTOR="${STORE}/runs/argotweak/qwen3vl_visual_selector_v1/seed20261501"
if [[ ! -s "${SELECTOR}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/train_argotweak_visual_selector.py \
    --features "${FEATURE_ROOT}/train" --output-dir "${SELECTOR}" \
    --seed 20261501 --device cuda >"${ROOT}/logs/selector_seed20261501.log" 2>&1
fi
VALIDATION="${STORE}/runs/argotweak/qwen3vl_visual_selector_v1/validation_seed20261501"
if [[ ! -s "${VALIDATION}/summary.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" scripts/evaluate_argotweak_visual_selector.py \
    --features "${FEATURE_ROOT}/val" --checkpoint "${SELECTOR}/best.pt" \
    --output-dir "${VALIDATION}" --device cuda >"${ROOT}/logs/selector_validation.log" 2>&1
fi

"${GIS_PYTHON}" -c \
  "import json,pathlib; p=pathlib.Path('${VALIDATION}/summary.json'); m=json.loads(p.read_text()); ok=m['utility_gain_over_direct']>0 and m['exact_top1']>=0.10; pathlib.Path('${ROOT}/status/PROMOTED' if ok else '${ROOT}/status/NOT_PROMOTED').touch(); print(json.dumps({'promoted':ok,'metrics':m},indent=2))" \
  >"${ROOT}/logs/promotion.json"
touch "${ROOT}/QUEUE_COMPLETED"
