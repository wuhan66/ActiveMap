#!/usr/bin/env bash
set -euo pipefail

ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ROOT}/envs/activemap-agent/bin/python"
RUN="${ROOT}/runs/sn7_active_catalog"
CAUSAL="${RUN}/final_tool_belief_causal_v1"
PLAN="${RUN}/plan_execute_qwen_seed20260718_n512"
PLAN_MANIFEST="${RUN}/plan_execute_qwen_seed20260718_n512_manifest.json"
OUTPUT="${RUN}/final_executable_controller_table_v1"
POLL_SECONDS="${POLL_SECONDS:-60}"

SFT="${RUN}/closed_loop_sft_seed20260718_n512_writeback/evaluation/writeback.jsonl"
REACT="${RUN}/react_style_qwen_seed20260718_n512_v2_writeback/evaluation/writeback.jsonl"
PLAN_WB="${RUN}/plan_execute_qwen_seed20260718_n512_writeback/evaluation/writeback.jsonl"
NO_TOOL="${CAUSAL}/seed20260718/no_tool_writeback/evaluation/writeback.jsonl"
FORCED="${CAUSAL}/seed20260718/forced_recurrent_writeback/evaluation/writeback.jsonl"
FROZEN="${CAUSAL}/seed20260718/selective_frozen_prior_writeback/evaluation/writeback.jsonl"
ACTIVE="${CAUSAL}/seed20260718/selective_writeback/evaluation/writeback.jsonl"

until [[ -s "${CAUSAL}/manifest.json" && -s "${PLAN_MANIFEST}" ]]; do
  sleep "${POLL_SECONDS}"
done
for path in "${SFT}" "${REACT}" "${PLAN_WB}" "${NO_TOOL}" "${FORCED}" "${FROZEN}" "${ACTIVE}"; do
  [[ -s "${path}" ]] || { echo "missing executable table input: ${path}" >&2; exit 3; }
done
[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing existing executable table: ${OUTPUT}" >&2
  exit 4
}

cd "${REPO}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
"${PYTHON}" scripts/build_sn7_executable_controller_table.py \
  "${OUTPUT}" --reference recurrent_sft --candidate active_map \
  --method "recurrent_sft=${SFT}" \
  --method "react=${REACT}" \
  --method "plan_execute=${PLAN_WB}" \
  --method "final_no_tool=${NO_TOOL}" \
  --method "forced_tools=${FORCED}" \
  --method "selective_frozen_prior=${FROZEN}" \
  --method "active_map=${ACTIVE}" \
  --repetitions 5000 --seed 20260729
