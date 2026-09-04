#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
UPDATER="${RUN}/../updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
GPU="${GPU:-2}"
SEED=20260719

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

run_writeback() {
  local method="$1"
  local source="$2"
  local protocol="$3"
  local input="${RUN}/${method}_seed${SEED}_n512_v2_writeback_input.jsonl"
  local output="${RUN}/${method}_seed${SEED}_n512_v2_writeback"

  [[ -s "${source}/evaluation/traces.jsonl" ]] || {
    echo "missing traces: ${source}" >&2
    return 3
  }
  if [[ ! -s "${input}" ]]; then
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${source}/evaluation/traces.jsonl" "${input}" --split val
  fi
  if [[ -s "${output}/evaluation/writeback.jsonl" ]]; then
    echo "writeback already complete: ${output}"
    return
  fi
  [[ ! -e "${output}" ]] || {
    echo "refusing partial writeback: ${output}" >&2
    return 4
  }
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${DATA}/closed_loop_v1/episodes_val.jsonl" \
    "${input}" "${output}" --gpu "${GPU}" --image-size 512 \
    --threshold 0.5 --protocol-name "${protocol}" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --split val --limit 512 --monitor-interval 5
}

run_writeback \
  closed_loop_sft \
  "${RUN}/closed_loop_sft_seed${SEED}_n512_v2" \
  sn7-direct-sft-v2-seed20260719-writeback
run_writeback \
  react_style_qwen \
  "${RUN}/react_style_qwen_seed${SEED}_n512_v2" \
  sn7-react-v2-seed20260719-writeback
