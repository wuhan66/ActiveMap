#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_v2_persistent_seed20260821}"
SFT_ROOT="${MUNO21_SFT_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821}"
SFT_ADAPTER="${MUNO21_SFT_ADAPTER:-${SFT_ROOT}/checkpoints/checkpoint-200}"
GPU="${MUNO21_AGENT_GPU:-3}"
LABEL="sft-checkpoint-200-quality-v2"
STATUS_PATH="${RUN_ROOT}/evaluation/sft_quality_refresh.exit_code"

mkdir -p "${RUN_ROOT}/evaluation"
rm -f "$STATUS_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS_PATH"' EXIT

while [[ ! -s "${RUN_ROOT}/evaluation/watcher.exit_code" ]] || \
      [[ "$(cat "${RUN_ROOT}/evaluation/watcher.exit_code" 2>/dev/null)" != "0" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for frozen DPO evaluation"
  sleep 15
done

while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}"
  sleep 15
done

cd "$PROJECT_ROOT"
bash scripts/evaluate_muno21_agent_run_adapter.sh \
  "$RUN_ROOT" "$SFT_ADAPTER" "$LABEL" "$GPU"

echo "[$(date --iso-8601=seconds)] refreshed frozen SFT quality-cost evaluation completed"
