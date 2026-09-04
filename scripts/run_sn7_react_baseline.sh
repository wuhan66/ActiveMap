#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final}"
TOOL_BELIEF_CHECKPOINT="${TOOL_BELIEF_CHECKPOINT:-${STORAGE_ROOT}/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt}"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
GPU="${GPU:-4}"
SEED="${SEED:-20260718}"
DRY_RUN="${DRY_RUN:-0}"
PROTOCOL_VERSION="${REACT_PROTOCOL_VERSION:-v2}"

case "${MODE}" in
  smoke)
    OUTPUT="${RUN}/react_style_qwen_seed${SEED}_smoke_${PROTOCOL_VERSION}"
    LIMIT=2
    BOOTSTRAP=0
    ;;
  full)
    OUTPUT="${RUN}/react_style_qwen_seed${SEED}_n512_${PROTOCOL_VERSION}"
    LIMIT=512
    BOOTSTRAP=500
    ;;
  *)
    echo "usage: $0 {smoke|full}" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

for path in \
  "${MODEL}" \
  "${ADAPTER}/adapter_config.json" \
  "${TOOL_BELIEF_CHECKPOINT}" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/closed_loop_v1/episodes_val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl"; do
  [[ -e "${path}" ]] || {
    echo "missing required ReAct input: ${path}" >&2
    exit 3
  }
done

[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing existing ReAct output: ${OUTPUT}" >&2
  exit 4
}

extra_args=()
if [[ "${DRY_RUN}" == "1" ]]; then
  extra_args+=(--dry-run)
fi

"${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
  "${MODEL}" "${ADAPTER}" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/closed_loop_v1/episodes_val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
  "${OUTPUT}" --gpu "${GPU}" --seed "${SEED}" \
  --policy-mode react --tool-mode model \
  --tool-belief-checkpoint "${TOOL_BELIEF_CHECKPOINT}" \
  --tool-artifact-root "${OUTPUT}/tool_artifacts" --tool-out-size 256 \
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
  --max-candidates 16 --max-acquisitions 2 --max-new-tokens 96 \
  --bootstrap-repetitions "${BOOTSTRAP}" --limit "${LIMIT}" \
  --monitor-interval 5 "${extra_args[@]}"
