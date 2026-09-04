#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-3}"
PROJECT_ROOT="/home/wh/projects/activemap-v1"
DATA_ROOT="/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/agent_data_v6_anonymized"
RUN_DIR="/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v4_no_acquire_oversample_seed20260821"

while pgrep -f "evaluate_muno21_agent_static_candidates.sh ${GPU}" >/dev/null \
  || [[ -n "$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}"
  sleep 30
done

echo "[$(date --iso-8601=seconds)] launching no-ACQUIRE-oversampling ablation"
env \
  ACTIVEMAP_AGENT_PYTHON=/home/wh/venvs/activemap/bin/python \
  ACTIVEMAP_AGENT_SITE_PACKAGES=/mnt/mydisk/wh/ActiveMap/envs/agent_peft_overlay \
  MUNO21_AGENT_GPU="${GPU}" \
  MUNO21_AGENT_DATA_ROOT="${DATA_ROOT}" \
  MUNO21_AGENT_TRAIN_FILE="${DATA_ROOT}/train/sft.jsonl" \
  MUNO21_AGENT_EVAL_FILE="${DATA_ROOT}/val/sft.jsonl" \
  MUNO21_AGENT_RUN_DIR="${RUN_DIR}" \
  "${PROJECT_ROOT}/scripts/start_muno21_agent_sft.sh"
