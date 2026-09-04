#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
if [[ -f "${PROJECT_ROOT}/scripts/server_hdpi_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
fi
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-${ACTIVEMAP_AGENT_ENV}/bin/python}"
AGENT_SITE_PACKAGES="${ACTIVEMAP_AGENT_SITE_PACKAGES:-}"
GPU="${MUNO21_AGENT_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-4B}"
DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v5_ensemble}"
RUN_DIR="${MUNO21_AGENT_RUN_DIR:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_sft_v1}"
TRAIN_FILE="${MUNO21_AGENT_TRAIN_FILE:-$DATA_ROOT/train/sft_balanced.jsonl}"
EVAL_FILE="${MUNO21_AGENT_EVAL_FILE:-$DATA_ROOT/val/sft.jsonl}"

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/assert_allowed_gpu.sh"
activemap_assert_allowed_gpu "${GPU}"
cd "$PROJECT_ROOT"
PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "$STORAGE_ROOT"

for path in "$MODEL/config.json" "$TRAIN_FILE" "$EVAL_FILE"; do
  [[ -s "$path" ]] || { echo "Required input is missing: $path" >&2; exit 1; }
done
mkdir -p "$RUN_DIR"
mkdir -p "$RUN_DIR/control"
PID_FILE="$RUN_DIR/train.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Agent SFT already running: PID $(cat "$PID_FILE")" >&2
  exit 1
fi
GPU_PIDS="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Physical GPU $GPU has active compute processes: $GPU_PIDS" >&2
  exit 1
fi
if [[ -f "$RUN_DIR/control/STOP" ]]; then
  echo "Remove $RUN_DIR/control/STOP before launching a new run" >&2
  exit 1
fi
PYTHONPATH="$AGENT_SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" -c "import accelerate, peft, tensorboard, torch, transformers" || {
  echo "Agent environment is incomplete; see environments/agent-requirements.txt" >&2
  exit 1
}

printf 'physical_gpu=%s\npython=%s\nmodel=%s\ndata_root=%s\ntrain_file=%s\neval_file=%s\n' \
  "$GPU" "$PYTHON" "$MODEL" "$DATA_ROOT" "$TRAIN_FILE" "$EVAL_FILE" \
  > "$RUN_DIR/launch.txt"
nohup env CUDA_VISIBLE_DEVICES="$GPU" \
  PYTHONPATH="$PROJECT_ROOT/src${AGENT_SITE_PACKAGES:+:$AGENT_SITE_PACKAGES}${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON" "$PROJECT_ROOT/scripts/train_agent_sft.py" \
  "$MODEL" "$TRAIN_FILE" "$RUN_DIR" \
  --eval-jsonl "$EVAL_FILE" \
  --epochs 2 --learning-rate 0.0002 --batch-size 1 \
  --gradient-accumulation 16 --max-length 2048 \
  --logging-steps 5 --eval-steps 100 --save-steps 100 \
  --lora-rank 16 --lora-alpha 32 --seed 20260821 \
  > "$RUN_DIR/train.log" 2>&1 < /dev/null &
pid=$!
echo "$pid" > "$PID_FILE"
sleep 3
if ! kill -0 "$pid" 2>/dev/null; then
  echo "Agent SFT failed to start; inspect $RUN_DIR/train.log" >&2
  exit 1
fi
echo "Agent SFT started: PID $pid, physical GPU $GPU, run $RUN_DIR"
