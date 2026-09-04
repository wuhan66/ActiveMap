#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/wh/projects/activemap-v1}"
STORAGE="${STORAGE:-/home/wh/ActiveMap}"
WRITEBACK="${WRITEBACK:-${STORAGE}/runs/muno21_evidence_value_writeback_val_20260725/value_seed3/writeback.jsonl}"
ROOT="${ROOT:-${STORAGE}/runs/muno21_graph_tolerance_ablation_20260725}"

launch() {
  local tolerance="$1"
  local label="${tolerance/./p}"
  local output="${ROOT}/tol_${label}"
  [[ ! -e "${output}" ]] || {
    echo "Refusing existing output: ${output}" >&2
    return 1
  }
  mkdir -p "${ROOT}/logs"
  (
    cd "${REPO}"
    MUNO21_GRAPH_SIMPLIFY_TOLERANCE="${tolerance}" nohup \
      bash scripts/run_muno21_official_graph_metrics.sh \
      "${WRITEBACK}" "${output}" \
      >"${ROOT}/logs/tol_${label}.log" 2>&1 < /dev/null &
    echo "$!" >"${ROOT}/tol_${label}.pid"
  )
  echo "tol_${label}: pid=$(cat "${ROOT}/tol_${label}.pid")"
}

queue_after() {
  local predecessor="$1"
  local tolerance="$2"
  local predecessor_label="${predecessor/./p}"
  mkdir -p "${ROOT}/logs"
  (
    while [[ ! -s "${ROOT}/tol_${predecessor_label}/exit_code.txt" ]]; do
      sleep 30
    done
    if [[ "$(cat "${ROOT}/tol_${predecessor_label}/exit_code.txt")" != 0 ]]; then
      echo "Predecessor tol_${predecessor_label} failed; queue stopped" >&2
      exit 1
    fi
    launch "${tolerance}"
  ) >"${ROOT}/logs/queue_${tolerance/./p}.log" 2>&1 < /dev/null &
  echo "$!" >"${ROOT}/queue_${tolerance/./p}.pid"
}

mkdir -p "${ROOT}"
launch 0.0
launch 0.5
launch 2.0
queue_after 0.0 3.0
