#!/usr/bin/env bash
set -euo pipefail

GPU="${1:?usage: queue_muno21_balanced_tool_assetfix.sh GPU SEED SELECTOR_SEED [SEED SELECTOR_SEED ...]}"
shift
[[ "$#" -ge 2 && $(( $# % 2 )) -eq 0 ]] || {
  echo "seed and selector seed arguments must be paired" >&2
  exit 2
}

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
LOG_ROOT="${STORAGE_ROOT}/logs/muno21_balanced_tool_assetfix_v1"
mkdir -p "${LOG_ROOT}"

while [[ "$#" -gt 0 ]]; do
  seed="$1"
  selector_seed="$2"
  shift 2
  run_dir="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed${seed}"
  rollout_root="${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v10_balanced_tool_assetfix_v1_seed${seed}_val"
  writeback_root="${run_dir}/evaluation/writeback_assetfix_v1"
  control="${run_dir}/evaluation/posttrain_assetfix_v1"
  prior_control="${run_dir}/evaluation/posttrain_watcher"
  echo "[$(date -Is)] queue seed=${seed} selector_seed=${selector_seed} gpu=${GPU}"
  while [[ ! -s "${prior_control}/COMPLETE.json" ]]; do
    if [[ -s "${prior_control}/FAILED.json" ]]; then
      echo "prior post-train evaluation failed for seed=${seed}" >&2
      exit 3
    fi
    sleep 60
  done
  MUNO21_AGENT_ROLLOUT_ROOT="${rollout_root}" \
  MUNO21_WRITEBACK_ROOT="${writeback_root}" \
  MUNO21_POSTTRAIN_CONTROL="${control}" \
  MUNO21_WAIT_FOR_GPU_FREE=1 \
    bash "${PROJECT_ROOT}/scripts/watch_muno21_balanced_tool_posttrain.sh" \
      "${GPU}" "${seed}" "${selector_seed}"
done

echo "[$(date -Is)] asset-fixed queue complete on gpu=${GPU}"
