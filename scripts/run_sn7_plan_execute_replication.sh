#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:?ADAPTER must identify the frozen SFT replication}"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
GPU="${GPU:?GPU is required}"
SEED="${SEED:?SEED is required}"
LIMIT="${LIMIT:-512}"
VERSION="${PLAN_EXECUTE_PROTOCOL_VERSION:-v2}"
OUTPUT="${RUN}/plan_execute_qwen_seed${SEED}_n${LIMIT}_${VERSION}"

for path in \
  "${MODEL}" \
  "${ADAPTER}/adapter_config.json" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/closed_loop_v1/episodes_val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl"; do
  [[ -e "${path}" ]] || {
    echo "missing plan-execute replication input: ${path}" >&2
    exit 3
  }
done
[[ ! -e "${OUTPUT}" ]] || { echo "refusing existing output: ${OUTPUT}" >&2; exit 4; }

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
  "${MODEL}" "${ADAPTER}" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/closed_loop_v1/episodes_val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
  "${OUTPUT}" --gpu "${GPU}" --seed "${SEED}" \
  --policy-mode plan_execute --tool-mode none --belief-mode recurrent \
  --max-candidates 16 --max-acquisitions 2 --max-new-tokens 96 \
  --bootstrap-repetitions 500 --limit "${LIMIT}" --monitor-interval 5
