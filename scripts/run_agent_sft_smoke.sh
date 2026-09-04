#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
SITE_PACKAGES="${ACTIVEMAP_AGENT_SITE_PACKAGES:-/mnt/mydisk/wh/ActiveMap/envs/agent_peft_overlay}"
GPU="${MUNO21_AGENT_GPU:-3}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-4B}"
DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/agent_data_v5_ensemble}"
RUN_DIR="${MUNO21_AGENT_SMOKE_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_smoke_assistant_only}"

GPU_PIDS="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ -n "${GPU_PIDS//[[:space:]]/}" ]]; then
  echo "Physical GPU $GPU has active compute processes: $GPU_PIDS" >&2
  exit 1
fi
mkdir -p "$RUN_DIR"
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$PROJECT_ROOT/src:$SITE_PACKAGES${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" scripts/train_agent_sft.py \
  "$MODEL" "$DATA_ROOT/train/sft_balanced.jsonl" "$RUN_DIR" \
  --eval-jsonl "$DATA_ROOT/val/sft.jsonl" \
  --epochs 1 --batch-size 1 --gradient-accumulation 16 \
  --max-train-samples 16 --max-eval-samples 8 --max-length 2048 \
  --logging-steps 1 --eval-steps 1 --save-steps 1 \
  > "$RUN_DIR/train.log" 2>&1
tail -n 80 "$RUN_DIR/train.log"
