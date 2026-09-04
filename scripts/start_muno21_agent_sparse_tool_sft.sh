#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
if [[ -n "${ACTIVEMAP_SERVER_ENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${ACTIVEMAP_SERVER_ENV}"
elif [[ -f "${PROJECT_ROOT}/scripts/server_hdpi_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
fi
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
AGENT_SITE_PACKAGES="${ACTIVEMAP_AGENT_SITE_PACKAGES:-}"
GPU="${MUNO21_AGENT_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
SEED="${MUNO21_AGENT_SEED:-20260821}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-4B}"
DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools}"
RUN_DIR="${MUNO21_AGENT_RUN_DIR:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_sparse_tool_sft_seed${SEED}}"
TRAINING_PROTOCOL="${MUNO21_AGENT_PROTOCOL:-natural-prior-sparse-grounded-tool-sft-v1}"
EARLY_STOPPING_PATIENCE="${MUNO21_AGENT_EARLY_STOPPING_PATIENCE:-2}"
SAVE_TOTAL_LIMIT="${MUNO21_AGENT_SAVE_TOTAL_LIMIT:-5}"
TRAIN_FILE="${DATA_ROOT}/train/sft_composed.jsonl"
EVAL_FILE="${DATA_ROOT}/val/sft_composed.jsonl"

[[ "${EARLY_STOPPING_PATIENCE}" =~ ^[0-9]+$ ]] || {
  echo "MUNO21_AGENT_EARLY_STOPPING_PATIENCE must be a nonnegative integer" >&2
  exit 2
}
[[ "${SAVE_TOTAL_LIMIT}" =~ ^[1-9][0-9]*$ ]] || {
  echo "MUNO21_AGENT_SAVE_TOTAL_LIMIT must be a positive integer" >&2
  exit 2
}

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/assert_allowed_gpu.sh"
activemap_assert_allowed_gpu "${GPU}"
cd "${PROJECT_ROOT}"
PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "${STORAGE_ROOT}"

for path in \
  "${MODEL}/config.json" \
  "${TRAIN_FILE}" \
  "${TRAIN_FILE%.jsonl}.summary.json" \
  "${EVAL_FILE}" \
  "${EVAL_FILE%.jsonl}.summary.json"; do
  [[ -s "${path}" ]] || { echo "Required input is missing: ${path}" >&2; exit 1; }
done

PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${AGENT_SITE_PACKAGES:+:${AGENT_SITE_PACKAGES}}" "${PYTHON}" - \
  "${TRAIN_FILE%.jsonl}.summary.json" "${EVAL_FILE%.jsonl}.summary.json" <<'PY'
import json
import sys
from pathlib import Path

train = json.loads(Path(sys.argv[1]).read_text())
val = json.loads(Path(sys.argv[2]).read_text())
assert train["split"] == "train" and train["training_oversampling"] is True
assert val["split"] == "val" and val["training_oversampling"] is False
assert train["action_counts"].get("USE_TOOL", 0) > 0
assert val["action_counts"].get("USE_TOOL", 0) > 0
assert train["test_assets_read"] is False and val["test_assets_read"] is False
PY

if [[ -e "${RUN_DIR}" ]]; then
  echo "Refusing to reuse run directory: ${RUN_DIR}" >&2
  exit 1
fi
GPU_PIDS="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Physical GPU ${GPU} has active compute processes: ${GPU_PIDS}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/control"
PYTHONPATH="${AGENT_SITE_PACKAGES}${AGENT_SITE_PACKAGES:+${PYTHONPATH:+:${PYTHONPATH}}}" \
  "${PYTHON}" -c "import accelerate, peft, tensorboard, torch, transformers"
printf 'protocol=%s\nphysical_gpu=%s\nseed=%s\npython=%s\nmodel=%s\ntrain_file=%s\neval_file=%s\nearly_stopping_patience=%s\nsave_total_limit=%s\n' \
  "${TRAINING_PROTOCOL}" "${GPU}" "${SEED}" "${PYTHON}" "${MODEL}" "${TRAIN_FILE}" "${EVAL_FILE}" \
  "${EARLY_STOPPING_PATIENCE}" "${SAVE_TOTAL_LIMIT}" \
  > "${RUN_DIR}/launch.txt"

nohup env CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${AGENT_SITE_PACKAGES:+:${AGENT_SITE_PACKAGES}}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -u "${PROJECT_ROOT}/scripts/train_agent_sft.py" \
  "${MODEL}" "${TRAIN_FILE}" "${RUN_DIR}" \
  --eval-jsonl "${EVAL_FILE}" \
  --epochs 4 --learning-rate 0.0002 --batch-size 1 \
  --gradient-accumulation 16 --max-length 2048 \
  --logging-steps 5 --eval-steps 100 --save-steps 100 \
  --save-total-limit "${SAVE_TOTAL_LIMIT}" \
  --early-stopping-patience "${EARLY_STOPPING_PATIENCE}" \
  --early-stopping-threshold 0.0005 \
  --lora-rank 16 --lora-alpha 32 --seed "${SEED}" \
  > "${RUN_DIR}/train.log" 2>&1 < /dev/null &
pid=$!
echo "${pid}" > "${RUN_DIR}/train.pid"
sleep 3
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "Sparse-tool SFT failed to start; inspect ${RUN_DIR}/train.log" >&2
  exit 1
fi
echo "Sparse-tool SFT started: PID ${pid}, physical GPU ${GPU}, run ${RUN_DIR}"
