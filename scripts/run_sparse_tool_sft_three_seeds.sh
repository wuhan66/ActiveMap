#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
export ACTIVEMAP_LAUNCHER_NAME="run_sparse_tool_sft_three_seeds.sh"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
SEED_LIST="${MUNO21_AGENT_SEEDS:-20260821 20260822 20260823}"
read -r -a seeds <<<"$SEED_LIST"

for seed in "${seeds[@]}"; do
  export MUNO21_AGENT_SEED="$seed"
  export MUNO21_AGENT_RUN_DIR="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_sparse_tool_sft_seed${seed}"
  promotion="${MUNO21_AGENT_RUN_DIR}/evaluation/selection/promoted_adapter.json"
  if [[ -s "$promotion" ]]; then
    echo "[$(date --iso-8601=seconds)] seed ${seed} already promoted; skipping"
    continue
  fi
  if [[ ! -s "${MUNO21_AGENT_RUN_DIR}/final/adapter_config.json" ]]; then
    bash "${PROJECT_ROOT}/scripts/start_muno21_agent_sparse_tool_sft.sh"
  fi
  if [[ -s "${MUNO21_AGENT_RUN_DIR}/train.pid" ]]; then
    train_pid="$(cat "${MUNO21_AGENT_RUN_DIR}/train.pid")"
    while kill -0 "$train_pid" 2>/dev/null; do
      echo "[$(date --iso-8601=seconds)] seed ${seed} training PID ${train_pid}"
      sleep 60
    done
  fi
  [[ -s "${MUNO21_AGENT_RUN_DIR}/final/adapter_config.json" ]] || {
    echo "SFT seed ${seed} failed before final adapter creation" >&2
    exit 4
  }
  bash "${PROJECT_ROOT}/scripts/evaluate_promote_sparse_tool_sft.sh"
done

echo "[$(date --iso-8601=seconds)] all sparse-tool SFT seeds promoted"
