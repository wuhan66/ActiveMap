#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUNS="${STORAGE_ROOT}/runs/agent/muno21_generated_oof_terminal_critic_4seed_v1"
BASELINE="${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v12_proactive_tool_four_seed_threshold009_v1"
OUTPUT="${STORAGE_ROOT}/artifacts/paper_evidence/muno21_generated_oof_terminal_critic_4seed_v1"
STATE="${STORAGE_ROOT}/logs/muno21_generated_oof_terminal_critic_4seed_v1/aggregate.state"

mkdir -p "$(dirname "${STATE}")"
echo "waiting_for_four_critics" > "${STATE}"
while true; do
  count="$(find "${RUNS}" -mindepth 2 -maxdepth 2 -name summary.json 2>/dev/null | wc -l)"
  [[ "${count}" -eq 4 ]] && break
  sleep 60
done
cd "${PROJECT_ROOT}"
echo "aggregating" > "${STATE}"
"${PYTHON}" scripts/summarize_muno21_generated_oof_critic.py \
  "${RUNS}" "${BASELINE}" "${OUTPUT}" --bootstrap 10000
echo "complete" > "${STATE}"
