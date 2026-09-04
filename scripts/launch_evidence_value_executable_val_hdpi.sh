#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/wh/projects/activemap-v1}"
ROOT="${ROOT:-/home/wh/ActiveMap/runs/evidence_value_executable_val_20260725}"
PYTHON="${PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
STATES="${STATES:-/home/wh/ActiveMap/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/states_val_step0.jsonl}"
EPISODES="${EPISODES:-/home/wh/ActiveMap/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl}"
UPDATER="${UPDATER:-/home/wh/ActiveMap/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt}"
TRACE_ROOT="${TRACE_ROOT:-/home/wh/ActiveMap/runs/evidence_value_closed_loop_val_20260725}"

declare -A TRACES=(
  [value_seed1]="${TRACE_ROOT}/seed1/value_seed1.jsonl"
  [value_seed2]="${TRACE_ROOT}/seed2/value_seed2.jsonl"
  [value_seed3]="${TRACE_ROOT}/seed3/value_seed3.jsonl"
  [utility_only]="${TRACE_ROOT}/matched_baselines/utility_only.jsonl"
)
declare -A GPUS=(
  [value_seed1]=0
  [value_seed2]=2
  [value_seed3]=3
  [utility_only]=6
)

if [[ -e "${ROOT}" ]]; then
  echo "Refusing to overwrite ${ROOT}" >&2
  exit 1
fi
mkdir -p "${ROOT}/inputs" "${ROOT}/logs" "${ROOT}/pids"
cd "${REPO}"

for label in value_seed1 value_seed2 value_seed3 utility_only; do
  "${PYTHON}" scripts/prepare_terminal_evidence_writeback.py \
    "${STATES}" "${TRACES[${label}]}" "${ROOT}/inputs/${label}_terminal.jsonl" \
    >"${ROOT}/logs/${label}_prepare.log"
  "${PYTHON}" scripts/apply_terminal_operation_gate.py \
    "${ROOT}/inputs/${label}_terminal.jsonl" \
    "${ROOT}/inputs/${label}_add_gate.jsonl" \
    --allow-operation ADD \
    >"${ROOT}/logs/${label}_gate.log"
done

for label in value_seed1 value_seed2 value_seed3 utility_only; do
  nohup "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" \
    "${EPISODES}" \
    "${ROOT}/inputs/${label}_add_gate.jsonl" \
    "${ROOT}/writebacks/${label}" \
    --gpu "${GPUS[${label}]}" \
    --python "${PYTHON}" \
    --image-size 512 \
    --threshold 0.5 \
    --delta-margin 0.0 \
    --protocol-name "sn7-evidence-value-last-add-gate-margin0-v1" \
    --split val \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap" \
    >"${ROOT}/logs/${label}_writeback_launcher.log" 2>&1 < /dev/null &
  echo "$!" >"${ROOT}/pids/${label}.pid"
  echo "${label}: pid=$(cat "${ROOT}/pids/${label}.pid") gpu=${GPUS[${label}]}"
done
