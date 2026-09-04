#!/usr/bin/env bash
set -euo pipefail

if [[ "${ACTIVEMAP_ALLOW_LEGACY_PIPELINE:-0}" != "1" ]]; then
  echo "Legacy v8-to-v9 single-seed queue is disabled on hdpi-sys1." >&2
  echo "Use run_sparse_tool_sft_three_seeds.sh for the frozen paper protocol." >&2
  exit 64
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
V8_RUN="${MUNO21_V8_RUN:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_sparse_tool_sft_seed20260821}"
V9_RUN="${MUNO21_V9_RUN:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_natural_sparse_tool_sft_seed20260821}"
V9_DATA="${MUNO21_V9_DATA:-/mnt/mydisk/wh/ActiveMap/processed/muno21_v2/agent/agent_data_v9_natural_sparse_tools}"
LOG_ROOT="${MUNO21_LOG_ROOT:-/mnt/mydisk/wh/ActiveMap/logs}"
GPU="${MUNO21_AGENT_GPU:-3}"
EVAL_GPU="${MUNO21_EARLY_EVAL_GPU:-2}"
STATUS="${V9_RUN}.queue_exit_code"

rm -f "$STATUS"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS"' EXIT

v8_pid="$(cat "${V8_RUN}/train.pid")"
while kill -0 "$v8_pid" 2>/dev/null; do
  echo "[$(date --iso-8601=seconds)] waiting for v8 training PID ${v8_pid}"
  sleep 60
done
while [[ ! -s "${V8_RUN}/evaluation/watcher.exit_code" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for v8 static decision"
  sleep 60
done

[[ "$(cat "${V8_RUN}/evaluation/watcher.exit_code")" == "0" ]] || {
  echo "v8 watcher failed; refusing to launch v9" >&2
  exit 2
}
[[ -s "${V8_RUN}/evaluation/selection/static_rejection.txt" ]] || {
  echo "v8 was not statically rejected; v9 is not launched"
  exit 0
}
[[ ! -e "$V9_RUN" ]] || {
  echo "v9 run already exists; refusing to reuse it" >&2
  exit 2
}
while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for physical GPU ${GPU}"
  sleep 30
done

cd "$PROJECT_ROOT"
env \
  ACTIVEMAP_AGENT_PYTHON=/home/wh/venvs/activemap/bin/python \
  MUNO21_AGENT_GPU="$GPU" \
  MUNO21_AGENT_DATA_ROOT="$V9_DATA" \
  MUNO21_AGENT_RUN_DIR="$V9_RUN" \
  bash scripts/start_muno21_agent_sparse_tool_sft.sh

setsid -f env \
  MUNO21_AGENT_GPU="$GPU" \
  MUNO21_AGENT_RUN_DIR="$V9_RUN" \
  MUNO21_TOOL_DATA="$V9_DATA" \
  bash scripts/watch_evaluate_sparse_tool_agent.sh \
  > "${LOG_ROOT}/watch_evaluate_sparse_tool_agent_v9.log" 2>&1 < /dev/null

for step in 100 200; do
  setsid -f env \
    MUNO21_EARLY_CHECKPOINT="$step" \
    MUNO21_EARLY_EVAL_GPU="$EVAL_GPU" \
    MUNO21_AGENT_RUN_DIR="$V9_RUN" \
    bash scripts/watch_evaluate_sparse_tool_checkpoint100.sh \
    > "${LOG_ROOT}/watch_evaluate_sparse_tool_v9_checkpoint${step}.log" 2>&1 < /dev/null
done

echo "[$(date --iso-8601=seconds)] v9 training and validation watchers launched"
