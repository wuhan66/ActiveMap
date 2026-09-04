#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
GPU="${MUNO21_SELECTOR_GPU:-3}"
DPO_RUN="${MUNO21_DPO_RUN_DIR:-/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_v2_persistent_seed20260821}"
LOG="${MUNO21_GENERIC_WATCH_LOG:-/mnt/mydisk/wh/ActiveMap/logs/watch_train_muno21_generic_selector_3seed.log}"

mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1

for status in \
  "$DPO_RUN/evaluation/watcher.exit_code" \
  "$DPO_RUN/evaluation/selection/paired_comparison.exit_code"; do
  while [[ ! -s "$status" ]]; do
    echo "[$(date --iso-8601=seconds)] waiting for $status"
    sleep 60
  done
  [[ "$(cat "$status")" == "0" ]] || {
    echo "Required DPO evaluation stage failed: $status=$(cat "$status")" >&2
    exit 2
  }
done

configs=(
  muno21_evidence_generic_v5_server.yaml
  muno21_evidence_generic_v5_seed2_server.yaml
  muno21_evidence_generic_v5_seed3_server.yaml
)
runs=(
  muno21_evidence_generic_v5_seed20260811
  muno21_evidence_generic_v5_seed20260812
  muno21_evidence_generic_v5_seed20260813
)

for index in "${!configs[@]}"; do
  config="$PROJECT_ROOT/configs/selector/${configs[$index]}"
  run_dir="/mnt/mydisk/wh/ActiveMap/runs/selector/${runs[$index]}"
  if [[ -s "$run_dir/best.pt" && "$(cat "$run_dir/exit_code.txt" 2>/dev/null || true)" == "0" ]]; then
    echo "[$(date --iso-8601=seconds)] already complete: $run_dir"
    continue
  fi
  while [[ -n "$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | tr -d '[:space:]')" ]]; do
    echo "[$(date --iso-8601=seconds)] waiting for physical GPU $GPU"
    sleep 30
  done
  echo "[$(date --iso-8601=seconds)] launching ${configs[$index]}"
  ACTIVEMAP_PYTHON="$PYTHON" \
    MUNO21_SELECTOR_GPU="$GPU" \
    MUNO21_SELECTOR_CONFIG="$config" \
    MUNO21_SELECTOR_RUN_DIR="$run_dir" \
    bash "$PROJECT_ROOT/scripts/start_muno21_evidence_selector.sh"
  pid="$(cat "$run_dir/train.pid")"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 30
  done
  [[ -s "$run_dir/best.pt" && "$(cat "$run_dir/exit_code.txt")" == "0" ]] || {
    echo "Generic selector failed: $run_dir" >&2
    exit 3
  }
done

echo "[$(date --iso-8601=seconds)] three-seed generic selector training complete"
