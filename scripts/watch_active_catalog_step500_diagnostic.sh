#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}"
DATA_ROOT="${DATA_ROOT:-/home/wh/ActiveMap/processed/sn7_v1/agent/sequential_selector_v1}"
SFT_ROOT="${SFT_ROOT:-${DATA_ROOT}/full/active_catalog_sft_v4}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
WEIGHTED_FAMILY="${WEIGHTED_FAMILY:-qwen3vl4b_seed1_eval500}"
UNWEIGHTED_FAMILY="${UNWEIGHTED_FAMILY:-qwen3vl4b_unweighted_seed1}"
SEED="${SEED:-20260717}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-500}"
WEIGHTED_GPU="${WEIGHTED_GPU:-4}"
UNWEIGHTED_GPU="${UNWEIGHTED_GPU:-5}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT_ROOT}"
source scripts/server_hdpi_env.sh
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "${WEIGHTED_GPU}"
activemap_assert_allowed_gpu "${UNWEIGHTED_GPU}"
[[ "${WEIGHTED_GPU}" != "${UNWEIGHTED_GPU}" ]] || {
  echo "diagnostic evaluations require distinct GPUs" >&2
  exit 2
}

weighted_seed="${RUN_ROOT}/${WEIGHTED_FAMILY}/seed${SEED}"
unweighted_seed="${RUN_ROOT}/${UNWEIGHTED_FAMILY}/seed${SEED}"
weighted_adapter="${weighted_seed}/checkpoints/checkpoint-${CHECKPOINT_STEP}"
unweighted_adapter="${unweighted_seed}/checkpoints/checkpoint-${CHECKPOINT_STEP}"
weighted_output="${weighted_seed}/active_catalog_val_step${CHECKPOINT_STEP}"
unweighted_output="${unweighted_seed}/active_catalog_val_step${CHECKPOINT_STEP}"
report="${RUN_ROOT}/seed1_sampling_ablation_step${CHECKPOINT_STEP}.json"

for adapter in "${weighted_adapter}" "${unweighted_adapter}"; do
  while [[ ! -s "${adapter}/adapter_config.json" ]]; do
    echo "$(date -Is) waiting for ${adapter}"
    sleep "${POLL_SECONDS}"
  done
done
for output in "${weighted_output}" "${unweighted_output}" "${report}"; do
  [[ ! -e "${output}" ]] || { echo "refusing existing diagnostic output: ${output}" >&2; exit 3; }
done
for gpu in "${WEIGHTED_GPU}" "${UNWEIGHTED_GPU}"; do
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} became occupied: ${pids}" >&2
    exit 4
  }
done

evaluate() {
  local gpu="$1"
  local adapter="$2"
  local output="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_active_catalog_selector.py \
    "${MODEL}" "${adapter}" "${SFT_ROOT}/val.jsonl" \
    "${SFT_ROOT}/val_evaluation_index.jsonl" "${output}" \
    --device cuda:0 --seed "${SEED}" --bootstrap-repetitions 2000
}

evaluate "${WEIGHTED_GPU}" "${weighted_adapter}" "${weighted_output}" &
weighted_pid=$!
evaluate "${UNWEIGHTED_GPU}" "${unweighted_adapter}" "${unweighted_output}" &
unweighted_pid=$!
failed=0
wait "${weighted_pid}" || failed=1
wait "${unweighted_pid}" || failed=1
(( failed == 0 )) || exit 5

"${PYTHON}" scripts/compare_active_catalog_sampling_ablation.py \
  "${weighted_output}/traces.jsonl" "${unweighted_output}/traces.jsonl" \
  "${report}" --repetitions 2000 --seed "${SEED}" \
  --checkpoint-step "${CHECKPOINT_STEP}" --diagnostic-only
