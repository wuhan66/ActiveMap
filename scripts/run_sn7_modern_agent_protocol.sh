#!/usr/bin/env bash
set -euo pipefail

PROTOCOL="${1:?usage: run_sn7_modern_agent_protocol.sh PROTOCOL MODE}"
MODE="${2:-smoke}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
ADAPTER="${ADAPTER:-${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final}"
TOOL_BELIEF_CHECKPOINT="${TOOL_BELIEF_CHECKPOINT:-${STORAGE_ROOT}/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt}"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
GPU="${GPU:-2}"
SEED="${SEED:-20260718}"
PROTOCOL_VERSION="${MODERN_AGENT_PROTOCOL_VERSION:-v3}"
ALLOW_SHARED_GPU="${ALLOW_SHARED_GPU:-0}"

case "${PROTOCOL}" in
  geommagent_style|sensesearch_style) ;;
  *) echo "unsupported protocol: ${PROTOCOL}" >&2; exit 2 ;;
esac
case "${MODE}" in
  smoke) LIMIT=2; BOOTSTRAP=0 ;;
  full) LIMIT=512; BOOTSTRAP=500 ;;
  *) echo "unsupported mode: ${MODE}" >&2; exit 2 ;;
esac

OUTPUT="${RUN}/${PROTOCOL}_qwen_seed${SEED}_${MODE}_${PROTOCOL_VERSION}"
for path in \
  "${MODEL}" \
  "${ADAPTER}/adapter_config.json" \
  "${TOOL_BELIEF_CHECKPOINT}" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/closed_loop_v1/episodes_val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl"; do
  [[ -e "${path}" ]] || { echo "missing protocol input: ${path}" >&2; exit 3; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "refusing existing output: ${OUTPUT}" >&2; exit 4; }

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
shared_gpu_args=()
if [[ "${ALLOW_SHARED_GPU}" == "1" ]]; then
  shared_gpu_args+=(--allow-shared-gpu)
fi
"${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
  "${MODEL}" "${ADAPTER}" \
  "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
  "${DATA}/closed_loop_v1/episodes_val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val.jsonl" \
  "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
  "${OUTPUT}" --gpu "${GPU}" --seed "${SEED}" \
  --policy-mode "${PROTOCOL}" --tool-mode model \
  --tool-belief-checkpoint "${TOOL_BELIEF_CHECKPOINT}" \
  --tool-artifact-root "${OUTPUT}/tool_artifacts" --tool-out-size 256 \
  --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
  --max-candidates 16 --max-acquisitions 2 --max-new-tokens 96 \
  --bootstrap-repetitions "${BOOTSTRAP}" --limit "${LIMIT}" \
  --monitor-interval 5 "${shared_gpu_args[@]}"
