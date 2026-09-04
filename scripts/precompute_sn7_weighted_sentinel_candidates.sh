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
GPU="${WEIGHTED_CANDIDATE_GPU:-5}"
FAMILY="${WEIGHTED_FAMILY:-qwen3vl4b_seed1_eval500}"

cd "$PROJECT_ROOT"
# shellcheck source=/dev/null
source scripts/server_hdpi_env.sh
# shellcheck source=/dev/null
source scripts/assert_allowed_gpu.sh
activemap_assert_allowed_gpu "$GPU"

sentinel="${SFT_ROOT}/balanced_sentinel_v1"
seed_root="${RUN_ROOT}/${FAMILY}/seed${SEED}"
candidate_root="${seed_root}/sentinel_checkpoint_selection"
mkdir -p "$candidate_root"

adapters=()
[[ -s "${seed_root}/diagnostic_adapters/checkpoint-1000/adapter_model.safetensors" ]] && \
  adapters+=("${seed_root}/diagnostic_adapters/checkpoint-1000")
while IFS= read -r adapter; do adapters+=("$adapter"); done < <(
  find "${seed_root}/checkpoints" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
)
adapters+=("${seed_root}/final")

ordinal=0
for adapter in "${adapters[@]}"; do
  [[ -s "${adapter}/adapter_model.safetensors" ]] || continue
  ordinal=$((ordinal + 1))
  output="${candidate_root}/candidate-${ordinal}-$(basename "$adapter")"
  [[ -s "${output}/summary.json" ]] && continue
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" scripts/evaluate_active_catalog_selector.py \
    "$MODEL" "$adapter" "${sentinel}/val.jsonl" \
    "${sentinel}/val_evaluation_index.jsonl" "$output" \
    --device cuda:0 --seed "$SEED" --bootstrap-repetitions 500
done
