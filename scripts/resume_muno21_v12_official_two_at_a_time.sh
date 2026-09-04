#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/wh/projects/activemap-v1}"
STORAGE="${STORAGE:-/home/wh/ActiveMap}"
SOURCE_ROOT="${SOURCE_ROOT:-${STORAGE}/runs/agent/muno21_v12_threshold009_writeback_v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${STORAGE}/runs/muno21_v12_official_three_seed_20260802}"
SEEDS=(20260822 20260823 20260824)
METHODS=(
  edit_conditioned_proactive_tools_safe_delta
  qwen3_4b_sft_calibrated_tool_to_belief_safe_delta
)

run_one() {
  local seed="$1"
  local method="$2"
  local name="seed${seed}_${method}"
  local input="${SOURCE_ROOT}/seed${seed}/${method}/writeback.jsonl"
  local output="${OUTPUT_ROOT}/seed${seed}/${method}"
  [[ -s "${input}" ]] || { printf 'missing_input\n' >"${OUTPUT_ROOT}/status/${name}.txt"; return 3; }
  printf 'running\n' >"${OUTPUT_ROOT}/status/${name}.txt"
  set +e
  (
    cd "${REPO}"
    MUNO21_EVAL_SPLIT=val MUNO21_GRAPH_SIMPLIFY_TOLERANCE=3.0 \
      bash scripts/run_muno21_official_graph_metrics.sh "${input}" "${output}"
  ) >>"${OUTPUT_ROOT}/logs/${name}.log" 2>&1
  local code="$?"
  set -e
  printf '%s\n' "${code}" >"${OUTPUT_ROOT}/status/${name}.txt"
  return "${code}"
}

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/status"
printf '%s\n' "$(date --iso-8601=seconds)" >"${OUTPUT_ROOT}/recovery_started_at.txt"

overall=0
for seed in "${SEEDS[@]}"; do
  pids=()
  for method in "${METHODS[@]}"; do
    run_one "${seed}" "${method}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      overall=1
    fi
  done
done

printf '%s\n' "${overall}" >"${OUTPUT_ROOT}/recovery_exit_code.txt"
printf '%s\n' "$(date --iso-8601=seconds)" >"${OUTPUT_ROOT}/recovery_finished_at.txt"
exit "${overall}"
