#!/usr/bin/env bash
set -euo pipefail

PROTOCOL="${1:?usage: run_sn7_modern_agent_writebacks.sh PROTOCOL}"
case "${PROTOCOL}" in
  geommagent_style|sensesearch_style) ;;
  *) echo "unsupported protocol: ${PROTOCOL}" >&2; exit 2 ;;
esac

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
UPDATER="${STORAGE_ROOT}/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716/best_quality.pt"
GPU="${GPU:?GPU is required}"
SEEDS="${SEEDS:-20260718,20260719}"
IFS=',' read -r -a seed_values <<<"${SEEDS}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

for seed in "${seed_values[@]}"; do
  source="${RUN}/${PROTOCOL}_qwen_seed${seed}_full_v3"
  input="${RUN}/${PROTOCOL}_qwen_seed${seed}_full_v3_writeback_input.jsonl"
  output="${RUN}/${PROTOCOL}_qwen_seed${seed}_full_v3_writeback"
  [[ -s "${source}/evaluation/traces.jsonl" ]] || {
    echo "missing traces: ${source}" >&2
    exit 3
  }
  if [[ ! -s "${input}" ]]; then
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${source}/evaluation/traces.jsonl" "${input}" --split val
  fi
  if [[ -s "${output}/evaluation/writeback.jsonl" ]]; then
    echo "writeback already complete: ${output}"
    continue
  fi
  [[ ! -e "${output}" ]] || {
    echo "refusing partial writeback: ${output}" >&2
    exit 4
  }
  "${PYTHON}" scripts/launch_active_catalog_writeback.py \
    "${UPDATER}" "${DATA}/closed_loop_v1/episodes_val.jsonl" \
    "${input}" "${output}" --gpu "${GPU}" --image-size 512 \
    --threshold 0.5 \
    --protocol-name "sn7-${PROTOCOL}-seed${seed}-writeback-v1" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --split val --limit 512 --monitor-interval 5
done
