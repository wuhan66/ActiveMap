#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog}"
DATA_ROOT="${DATA_ROOT:-${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1}"
SFT_ROOT="${SFT_ROOT:-${DATA_ROOT}/full/active_catalog_sft_v4}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
SEED="${SEED:-20260717}"
CHECKPOINT_STEP="${CHECKPOINT_STEP:-1000}"
GPU="${SENTINEL_GPU:-6}"
POLL_SECONDS="${POLL_SECONDS:-60}"
WEIGHTED_FAMILY="${WEIGHTED_FAMILY:-qwen3vl4b_seed1_eval500}"
UNWEIGHTED_FAMILY="${UNWEIGHTED_FAMILY:-qwen3vl4b_unweighted_seed1}"

cd "$PROJECT_ROOT"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "$GPU"

sentinel="${SFT_ROOT}/balanced_sentinel_v1"
if [[ ! -s "${sentinel}/summary.json" ]]; then
  "$PYTHON" scripts/build_active_catalog_balanced_sentinel.py \
    "${SFT_ROOT}/val.jsonl" "${SFT_ROOT}/val_evaluation_index.jsonl" \
    "$sentinel" --seed "$SEED"
fi
if ! grep -q '"image_assets"' "${sentinel}/summary.json"; then
  "$PYTHON" scripts/build_active_catalog_balanced_sentinel.py \
    "${SFT_ROOT}/val.jsonl" "${SFT_ROOT}/val_evaluation_index.jsonl" \
    "$sentinel" --seed "$SEED" --repair-images
fi

resolve_adapter() {
  local seed_root="$1"
  local checkpoint="${seed_root}/checkpoints/checkpoint-${CHECKPOINT_STEP}"
  local pinned="${seed_root}/diagnostic_adapters/checkpoint-${CHECKPOINT_STEP}"
  while true; do
    if [[ -s "${pinned}/adapter_model.safetensors" ]]; then
      printf '%s\n' "$pinned"
      return
    fi
    if [[ -s "${checkpoint}/adapter_model.safetensors" ]]; then
      printf '%s\n' "$checkpoint"
      return
    fi
    echo "$(date -Is) waiting for checkpoint or pinned diagnostic adapter under ${seed_root}" >&2
    sleep "$POLL_SECONDS"
  done
}

weighted_seed_root="${RUN_ROOT}/${WEIGHTED_FAMILY}/seed${SEED}"
unweighted_seed_root="${RUN_ROOT}/${UNWEIGHTED_FAMILY}/seed${SEED}"
weighted_adapter="$(resolve_adapter "$weighted_seed_root")"
unweighted_adapter="$(resolve_adapter "$unweighted_seed_root")"

pin_adapter() {
  local source="$1"
  local seed_root="$2"
  local pinned="${seed_root}/diagnostic_adapters/checkpoint-${CHECKPOINT_STEP}"
  if [[ ! -s "${pinned}/adapter_model.safetensors" ]]; then
    mkdir -p "${seed_root}/diagnostic_adapters"
    [[ ! -e "$pinned" ]] || { echo "incomplete pinned adapter: ${pinned}" >&2; exit 5; }
    mkdir "$pinned"
    cp -a "${source}/adapter_config.json" "${source}/adapter_model.safetensors" "$pinned/"
  fi
  [[ -s "${pinned}/adapter_config.json" && -s "${pinned}/adapter_model.safetensors" ]] || {
    echo "failed to pin adapter: ${pinned}" >&2
    exit 5
  }
  printf '%s\n' "$pinned"
}

weighted_adapter="$(pin_adapter "$weighted_adapter" "$weighted_seed_root")"
unweighted_adapter="$(pin_adapter "$unweighted_adapter" "$unweighted_seed_root")"

while screen -ls 2>/dev/null | grep -q activemap_muno21_replicates_after_sn7_step500; do
  echo "$(date -Is) waiting for MUNO21 replicate training/evaluation to release its two-GPU allocation"
  sleep "$POLL_SECONDS"
done
while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "$(date -Is) waiting for GPU ${GPU}"
  sleep "$POLL_SECONDS"
done

weighted_output="${RUN_ROOT}/${WEIGHTED_FAMILY}/seed${SEED}/active_catalog_val_step${CHECKPOINT_STEP}_sentinel"
unweighted_output="${RUN_ROOT}/${UNWEIGHTED_FAMILY}/seed${SEED}/active_catalog_val_step${CHECKPOINT_STEP}_sentinel"
report="${RUN_ROOT}/seed1_sampling_ablation_step${CHECKPOINT_STEP}_sentinel.json"

evaluate() {
  local adapter="$1"
  local output="$2"
  if [[ ! -s "${output}/summary.json" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/evaluate_active_catalog_selector.py \
      "$MODEL" "$adapter" "${sentinel}/val.jsonl" \
      "${sentinel}/val_evaluation_index.jsonl" "$output" \
      --device cuda:0 --seed "$SEED" --bootstrap-repetitions 500
  fi
}

evaluate "$weighted_adapter" "$weighted_output"
evaluate "$unweighted_adapter" "$unweighted_output"
if [[ ! -s "$report" ]]; then
  "$PYTHON" scripts/compare_active_catalog_sampling_ablation.py \
    "${weighted_output}/traces.jsonl" "${unweighted_output}/traces.jsonl" \
    "$report" --repetitions 1000 --seed "$SEED" \
    --checkpoint-step "$CHECKPOINT_STEP" --diagnostic-only
fi
