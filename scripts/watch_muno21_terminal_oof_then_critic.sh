#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
GENERATED="${STORAGE_ROOT}/runs/agent_eval/qwen3_vl_4b_terminal_oof_predictions_v1"
STATE="${STORAGE_ROOT}/logs/qwen3_vl_4b_terminal_oof_predictions_v1/watcher.state"

cd "${PROJECT_ROOT}"
echo "waiting_for_generated_oof" > "${STATE}"
while true; do
  complete=0
  for fold in 0 1 2; do
    [[ -s "${GENERATED}/fold${fold}/summary.json" ]] && complete=$((complete + 1))
  done
  [[ "${complete}" -eq 3 ]] && break
  sleep 60
done
echo "launching_generated_oof_critic" > "${STATE}"
bash scripts/launch_muno21_generated_oof_critic_4seed.sh
echo "critic_launched" > "${STATE}"
