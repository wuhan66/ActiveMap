#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
AGENT_SITE_PACKAGES="${ACTIVEMAP_AGENT_SITE_PACKAGES:-/mnt/mydisk/wh/ActiveMap/envs/agent_peft_overlay}"
GPU="${MUNO21_EARLY_EVAL_GPU:-2}"
STEP="${MUNO21_EARLY_CHECKPOINT:-100}"
RUN_DIR="${MUNO21_AGENT_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sparse_tool_sft_seed20260821}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-4B}"
DATA="/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/agent_data_v8_sparse_tools/val/sft_composed.jsonl"
ADAPTER="${RUN_DIR}/checkpoints/checkpoint-${STEP}"
OUTPUT="${RUN_DIR}/evaluation/checkpoint-${STEP}/actions"
STATUS="${RUN_DIR}/evaluation/checkpoint-${STEP}/early_eval.exit_code"

mkdir -p "$(dirname "$STATUS")"
rm -f "$STATUS"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS"' EXIT

while [[ ! -s "${ADAPTER}/adapter_model.safetensors" || ! -s "${ADAPTER}/trainer_state.json" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for checkpoint-${STEP}"
  sleep 30
done
while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}"
  sleep 30
done

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="${PROJECT_ROOT}/src:${AGENT_SITE_PACKAGES}${PYTHONPATH:+:${PYTHONPATH}}"
"$PYTHON" scripts/evaluate_agent_actions.py \
  "$MODEL" "$DATA" "$OUTPUT" \
  --adapter "$ADAPTER" --device cuda --batch-size 2 \
  --max-length 2048 --max-new-tokens 96
echo "[$(date --iso-8601=seconds)] checkpoint-${STEP} early static evaluation complete"
