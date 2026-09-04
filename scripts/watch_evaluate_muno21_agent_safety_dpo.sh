#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_seed20260821}"
GPU="${MUNO21_AGENT_GPU:-3}"
LABEL="safety-dpo-best"
STATUS_PATH="${RUN_ROOT}/evaluation/watcher.exit_code"

mkdir -p "${RUN_ROOT}/evaluation"
rm -f "$STATUS_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS_PATH"' EXIT

while [[ ! -s "${RUN_ROOT}/train.pid" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for DPO train PID"
  sleep 30
done
train_pid="$(cat "${RUN_ROOT}/train.pid")"
while kill -0 "$train_pid" 2>/dev/null; do
  echo "[$(date --iso-8601=seconds)] waiting for DPO PID ${train_pid}"
  sleep 60
done

for path in \
  "${RUN_ROOT}/final/adapter_model.safetensors" \
  "${RUN_ROOT}/train_metrics.json" \
  "${RUN_ROOT}/eval_metrics.json"; do
  [[ -s "$path" ]] || { echo "DPO ended without required artifact: $path" >&2; exit 2; }
done

while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}"
  sleep 30
done

cd "$PROJECT_ROOT"
bash scripts/evaluate_muno21_agent_run_adapter.sh \
  "$RUN_ROOT" "${RUN_ROOT}/final" "$LABEL" "$GPU"

PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  /home/wh/venvs/activemap/bin/python scripts/summarize_agent_checkpoints.py \
  "${RUN_ROOT}/evaluation" "${RUN_ROOT}/evaluation/selection" \
  --labels "$LABEL"

echo "[$(date --iso-8601=seconds)] formal safety DPO evaluation completed"
