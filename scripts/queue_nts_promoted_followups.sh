#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORE="${ACTIVEMAP_STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
ROOT="${STORE}/runs/queues/nts_promoted_followups_20260804"
mkdir -p "${ROOT}/logs" "${ROOT}/status"
exec 9>"${ROOT}/.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/QUEUE_COMPLETED" ]] || exit 0
cd "${PROJECT_ROOT}"

bash scripts/launch_muno21_qwen8b_promoted_replication_nts.sh \
  >"${ROOT}/logs/qwen8b_replication.log" 2>&1 &
pid_replication=$!

env GPU_NO_BELIEF=2 GPU_RECURRENT=3 MUNO21_V12_TOOL_NEED_THRESHOLD=0.05 \
  MUNO21_SCREEN_LOG_TAG=threshold005_causal \
  MUNO21_SCREEN_ROOT="${STORE}/artifacts/paper_rollouts/qwen3_8b_structured_acquisition_matched_v3_threshold005" \
  bash scripts/launch_muno21_qwen8b_structured_acquisition_screen_nts.sh \
  >"${ROOT}/logs/qwen8b_gate005_causal.log" 2>&1 &
pid_causal=$!

printf '{"gpu0_1":"qwen8b_sft_replication","gpu2_3":"gate005_belief_causal","status":"running"}\n' \
  >"${ROOT}/status/running.json"
status=0
wait "${pid_replication}" || status=1
wait "${pid_causal}" || status=1
if [[ "${status}" -eq 0 ]]; then
  touch "${ROOT}/QUEUE_COMPLETED"
  printf '{"status":"complete","test_assets_read":false}\n' >"${ROOT}/status/final.json"
else
  touch "${ROOT}/QUEUE_FAILED"
  printf '{"status":"failed","test_assets_read":false}\n' >"${ROOT}/status/final.json"
fi
exit "${status}"
