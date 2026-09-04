#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
AGENT_SITE_PACKAGES="${ACTIVEMAP_AGENT_SITE_PACKAGES:-}"
GPU="${MUNO21_AGENT_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
SEED="${MUNO21_AGENT_SEED:?Set MUNO21_AGENT_SEED}"
MODEL="${MUNO21_AGENT_MODEL:-${ACTIVEMAP_MODEL_ROOT}/Qwen3-4B}"
DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools}"
RUN_DIR="${MUNO21_AGENT_RUN_DIR:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_sparse_tool_sft_seed${SEED}}"
EVAL_FILE="${DATA_ROOT}/val/sft_composed.jsonl"
EVALUATION="${RUN_DIR}/evaluation"
SELECTION="${EVALUATION}/selection"
PROMOTION="${SELECTION}/promoted_adapter.json"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/assert_allowed_gpu.sh"
activemap_assert_allowed_gpu "${GPU}"
if [[ -s "$PROMOTION" ]]; then
  echo "SFT seed ${SEED} is already promoted: ${PROMOTION}"
  exit 0
fi
if [[ -s "${RUN_DIR}/train.pid" ]]; then
  train_pid="$(cat "${RUN_DIR}/train.pid")"
  while kill -0 "$train_pid" 2>/dev/null; do
    echo "[$(date --iso-8601=seconds)] waiting for SFT seed ${SEED}, PID ${train_pid}"
    sleep 60
  done
fi
[[ -s "${RUN_DIR}/final/adapter_config.json" ]] || {
  echo "SFT seed ${SEED} ended without a final adapter" >&2
  exit 3
}
[[ -s "${RUN_DIR}/eval_metrics.json" ]] || {
  echo "SFT seed ${SEED} ended without final validation metrics" >&2
  exit 3
}
[[ -s "$EVAL_FILE" ]] || { echo "Missing validation SFT data: $EVAL_FILE" >&2; exit 3; }

mkdir -p "$SELECTION"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${AGENT_SITE_PACKAGES:+:${AGENT_SITE_PACKAGES}}${PYTHONPATH:+:${PYTHONPATH}}"

mapfile -t adapters < <(
  find "${RUN_DIR}/checkpoints" -mindepth 1 -maxdepth 1 \
    -type d -name 'checkpoint-*' 2>/dev/null | sort -V
)
if [[ "${#adapters[@]}" -eq 0 ]]; then
  adapters=("${RUN_DIR}/final")
fi
labels=()
for adapter in "${adapters[@]}"; do
  label="$(basename "$adapter")"
  labels+=("$label")
  output="${EVALUATION}/${label}/actions"
  if [[ ! -s "${output}/summary.json" ]]; then
    "$PYTHON" scripts/evaluate_agent_actions.py \
      "$MODEL" "$EVAL_FILE" "$output" \
      --adapter "$adapter" --device cuda --batch-size 2 \
      --max-length 2048 --max-new-tokens 96
  fi
done

labels_csv="$(IFS=,; echo "${labels[*]}")"
decision="${SELECTION}/static_checkpoint_decision.json"
"$PYTHON" scripts/select_sparse_tool_sft_checkpoint.py \
  "$EVALUATION" "$decision" --labels "$labels_csv"
"$PYTHON" scripts/promote_sparse_tool_sft_adapter.py \
  "$RUN_DIR" "$decision" "$PROMOTION" --seed "$SEED"
echo "SFT seed ${SEED} validation promotion complete: ${PROMOTION}"
