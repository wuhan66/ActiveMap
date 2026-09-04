#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
GPU="${MUNO21_AGENT_GPU:-3}"
QUEUE_STATUS="${V4_QUEUE_STATUS:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v4_no_acquire_oversample_seed20260821/evaluation/queue.exit_code}"
BASE_ADAPTER="${MUNO21_DPO_MODEL:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821/checkpoints/checkpoint-200}"
RUN_DIR="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_seed20260821}"
STATUS_PATH="${RUN_DIR}.watcher.exit_code"

rm -f "$STATUS_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS_PATH"' EXIT

while [[ ! -s "$QUEUE_STATUS" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for v4 evaluation queue"
  sleep 30
done
if [[ "$(tr -d '[:space:]' < "$QUEUE_STATUS")" != "0" ]]; then
  echo "v4 evaluation queue failed; refusing to launch DPO" >&2
  exit 2
fi

for path in "$BASE_ADAPTER/adapter_model.safetensors" "$BASE_ADAPTER/adapter_config.json"; do
  [[ -s "$path" ]] || { echo "Missing frozen SFT adapter input: $path" >&2; exit 3; }
done
if [[ -s "$RUN_DIR/final/adapter_model.safetensors" ]]; then
  echo "Formal DPO run is already complete: $RUN_DIR"
  exit 0
fi

while true; do
  active_training="$(pgrep -u "$USER" -f '[t]rain_agent_(sft|dpo).py|[a]ctivemap.cli train-updater|[t]rain_tool_belief_spatial_decision_head.py' || true)"
  gpu_pids="$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -z "${active_training//[[:space:]]/}" && -z "${gpu_pids//[[:space:]]/}" ]]; then
    break
  fi
  echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU} and training slot"
  sleep 30
done

export ACTIVEMAP_AGENT_PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
export MUNO21_AGENT_GPU="$GPU"
export MUNO21_DPO_MODEL="$BASE_ADAPTER"
export MUNO21_DPO_RUN_DIR="$RUN_DIR"
cd "$PROJECT_ROOT"
bash scripts/start_muno21_agent_safety_dpo.sh

echo "[$(date --iso-8601=seconds)] formal safety DPO launch completed"
