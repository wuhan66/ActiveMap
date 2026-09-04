#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/wh/projects/activemap-v1}"
STORAGE="${STORAGE:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE}/envs/activemap-agent/bin/python}"
UPDATER="${UPDATER:-${STORAGE}/models/frozen_updater/muno21_v4_seed20260726/best_val_loss.pt}"
EPISODES="${EPISODES:-${STORAGE}/processed/muno21_v2/agent/episodes_train_val_v1.jsonl}"
STATES="${STATES:-${STORAGE}/processed/muno21_v2/agent/executable_value_v1/states_train_val_executable_balanced_512.jsonl}"
TRACE_ROOT="${TRACE_ROOT:-${STORAGE}/runs/muno21_evidence_value_closed_loop_val_20260725}"
WRITEBACK_ROOT="${WRITEBACK_ROOT:-${STORAGE}/runs/muno21_evidence_value_writeback_val_20260725}"
OFFICIAL_ROOT="${OFFICIAL_ROOT:-${STORAGE}/runs/muno21_graph_tolerance3_three_seed_20260725}"
SEED1_GPU="${SEED1_GPU:-0}"
SEED2_GPU="${SEED2_GPU:-2}"

require_file() {
  [[ -s "$1" ]] || { echo "Required input is missing: $1" >&2; exit 3; }
}

require_free_gpu() {
  local gpu="$1"
  local pids
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    echo "GPU ${gpu} has active compute processes: ${pids}" >&2
    exit 4
  fi
}

run_seed() {
  local seed="$1"
  local gpu="$2"
  local trace="${TRACE_ROOT}/value_seed${seed}.jsonl"
  local terminal_trace="${TRACE_ROOT}/value_seed${seed}_writeback.jsonl"
  local output="${WRITEBACK_ROOT}/value_seed${seed}"
  local official="${OFFICIAL_ROOT}/seed${seed}"

  require_file "${trace}"
  require_file "${STATES}"
  require_free_gpu "${gpu}"
  cd "${REPO}"
  if [[ ! -s "${terminal_trace}" ]]; then
    PYTHONPATH=src:. "${PYTHON}" scripts/prepare_terminal_evidence_writeback.py \
      "${STATES}" "${trace}" "${terminal_trace}"
  fi
  require_file "${terminal_trace}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=src:. "${PYTHON}" \
    scripts/evaluate_agent_map_writeback.py \
    "${UPDATER}" "${EPISODES}" "${terminal_trace}" "${output}" \
    --device cuda:0 \
    --split val \
    --image-size 512 \
    --threshold 0.5 \
    --simplify-tolerance 0.0 \
    --min-delta-component-pixels 0 \
    --delta-margin 0.0 \
    --confidence-floor 0.0 \
    --evidence-fusion confidence_weighted \
    --protocol-name muno21-road-footprint-vector-delta-v1 \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE}"
  require_file "${output}/writeback.jsonl"
  require_file "${output}/summary.json"
  MUNO21_GRAPH_SIMPLIFY_TOLERANCE=3.0 \
    bash scripts/run_muno21_official_graph_metrics.sh \
    "${output}/writeback.jsonl" "${official}"
  echo 0 >"${OFFICIAL_ROOT}/seed${seed}_pipeline_exit_code.txt"
}

if [[ "${1:-}" == "worker" ]]; then
  run_seed "${2:?missing seed}" "${3:?missing GPU}"
  exit 0
fi

for path in "${UPDATER}" "${EPISODES}" "${STATES}"; do
  require_file "${path}"
done
[[ "${SEED1_GPU}" != "${SEED2_GPU}" ]] || {
  echo "Seed GPUs must be distinct" >&2
  exit 7
}
[[ ! -e "${OFFICIAL_ROOT}" ]] || {
  echo "Refusing existing pipeline root: ${OFFICIAL_ROOT}" >&2
  exit 8
}

mkdir -p "${OFFICIAL_ROOT}/logs" "${OFFICIAL_ROOT}/pids"
cat >"${OFFICIAL_ROOT}/protocol.txt" <<EOF
split=val
test_assets_read=false
graph_simplify_tolerance=3.0
seed3_official_reuse=${STORAGE}/runs/muno21_graph_tolerance_ablation_20260725/tol_3p0
EOF

for pair in "1:${SEED1_GPU}" "2:${SEED2_GPU}"; do
  seed="${pair%%:*}"
  gpu="${pair##*:}"
  trace="${TRACE_ROOT}/value_seed${seed}.jsonl"
  output="${WRITEBACK_ROOT}/value_seed${seed}"
  official="${OFFICIAL_ROOT}/seed${seed}"
  require_file "${trace}"
  [[ ! -e "${output}" ]] || {
    echo "Refusing existing writeback output: ${output}" >&2
    exit 5
  }
  [[ ! -e "${official}" ]] || {
    echo "Refusing existing official output: ${official}" >&2
    exit 6
  }
  require_free_gpu "${gpu}"
done

for pair in "1:${SEED1_GPU}" "2:${SEED2_GPU}"; do
  seed="${pair%%:*}"
  gpu="${pair##*:}"
  REPO="${REPO}" STORAGE="${STORAGE}" PYTHON="${PYTHON}" UPDATER="${UPDATER}" \
    EPISODES="${EPISODES}" STATES="${STATES}" TRACE_ROOT="${TRACE_ROOT}" \
    WRITEBACK_ROOT="${WRITEBACK_ROOT}" OFFICIAL_ROOT="${OFFICIAL_ROOT}" \
    nohup bash "$0" worker "${seed}" "${gpu}" \
    >"${OFFICIAL_ROOT}/logs/seed${seed}_pipeline.log" 2>&1 < /dev/null &
  echo "$!" >"${OFFICIAL_ROOT}/pids/seed${seed}_pipeline.pid"
  echo "seed${seed}: gpu=${gpu} pipeline_pid=$(cat "${OFFICIAL_ROOT}/pids/seed${seed}_pipeline.pid")"
done
