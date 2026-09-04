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

require_file() {
  [[ -s "$1" ]] || { echo "Required input is missing: $1" >&2; exit 3; }
}

run_job() {
  local seed="$1"
  local method="$2"
  local input="${SOURCE_ROOT}/seed${seed}/${method}/writeback.jsonl"
  local output="${OUTPUT_ROOT}/seed${seed}/${method}"
  require_file "${input}"
  [[ ! -e "${output}" ]] || { echo "Refusing existing output: ${output}" >&2; exit 4; }
  mkdir -p "$(dirname "${output}")"
  cd "${REPO}"
  MUNO21_EVAL_SPLIT=val MUNO21_GRAPH_SIMPLIFY_TOLERANCE=3.0 \
    bash scripts/run_muno21_official_graph_metrics.sh "${input}" "${output}"
}

if [[ "${1:-}" == worker ]]; then
  seed="${2:?missing seed}"
  method="${3:?missing method}"
  name="seed${seed}_${method}"
  set +e
  run_job "${seed}" "${method}"
  code="$?"
  set -e
  printf '%s\n' "${code}" >"${OUTPUT_ROOT}/status/${name}.txt"
  exit "${code}"
fi

[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Refusing existing root: ${OUTPUT_ROOT}" >&2; exit 5; }
mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/pids" "${OUTPUT_ROOT}/status"
cat >"${OUTPUT_ROOT}/protocol.txt" <<EOF
split=val
test_assets_read=false
seeds=20260822,20260823,20260824
methods=edit_conditioned_proactive_tools_safe_delta,qwen3_4b_sft_calibrated_tool_to_belief_safe_delta
budgets=1.5,3.0,4.5
graph_simplify_tolerance=3.0
source_root=${SOURCE_ROOT}
EOF

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    require_file "${SOURCE_ROOT}/seed${seed}/${method}/writeback.jsonl"
  done
done

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
    name="seed${seed}_${method}"
    nohup bash "$0" worker "${seed}" "${method}" \
      >"${OUTPUT_ROOT}/logs/${name}.log" 2>&1 < /dev/null &
    pid="$!"
    echo "${pid}" >"${OUTPUT_ROOT}/pids/${name}.pid"
    printf 'running\n' >"${OUTPUT_ROOT}/status/${name}.txt"
    echo "${name}: pid=${pid}"
  done
done
