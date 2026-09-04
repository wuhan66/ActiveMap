#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
VERSION="${VERSION:-v1}"
OUT_SIZE="${OUT_SIZE:-256}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"

RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA_ROOT="${STORAGE_ROOT}/processed/sn7_v1/agent"
OUTPUT_ROOT="${DATA_ROOT}/step0_grounded_tool_pairs_seed30_${VERSION}"
LOG_ROOT="${STORAGE_ROOT}/logs"

TRAIN_TRANSITIONS="${RUN_ROOT}/step0_joint_train_seed30_v1/joint_transitions.jsonl"
VAL_TRANSITIONS="${RUN_ROOT}/step0_joint_val_seed30_v1/joint_transitions.jsonl"
TRAIN_EPISODES="${DATA_ROOT}/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl"
VAL_EPISODES="${DATA_ROOT}/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
ASSET_MAP="/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"

for input in \
  "${TRAIN_TRANSITIONS}" \
  "${VAL_TRANSITIONS}" \
  "${TRAIN_EPISODES}" \
  "${VAL_EPISODES}"; do
  test -f "${input}" || {
    echo "missing input: ${input}" >&2
    exit 2
  }
done

test ! -e "${OUTPUT_ROOT}/train" || {
  echo "refusing to overwrite ${OUTPUT_ROOT}/train" >&2
  exit 3
}
test ! -e "${OUTPUT_ROOT}/val" || {
  echo "refusing to overwrite ${OUTPUT_ROOT}/val" >&2
  exit 3
}

mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"
cd "${PROJECT_ROOT}"

"${PYTHON}" scripts/build_active_catalog_grounded_tool_data.py \
  "${TRAIN_TRANSITIONS}" \
  "${TRAIN_EPISODES}" \
  "${OUTPUT_ROOT}/train" \
  --split train \
  --out-size "${OUT_SIZE}" \
  --label-smoothing "${LABEL_SMOOTHING}" \
  --asset-root-map "${ASSET_MAP}" \
  >"${LOG_ROOT}/sn7_step0_grounded_tools_train_${VERSION}.log" 2>&1 &
train_pid=$!

"${PYTHON}" scripts/build_active_catalog_grounded_tool_data.py \
  "${VAL_TRANSITIONS}" \
  "${VAL_EPISODES}" \
  "${OUTPUT_ROOT}/val" \
  --split val \
  --out-size "${OUT_SIZE}" \
  --label-smoothing "${LABEL_SMOOTHING}" \
  --asset-root-map "${ASSET_MAP}" \
  >"${LOG_ROOT}/sn7_step0_grounded_tools_val_${VERSION}.log" 2>&1 &
val_pid=$!

status=0
wait "${train_pid}" || status=$?
wait "${val_pid}" || status=$?

if (( status != 0 )); then
  echo "grounded tool construction failed; inspect ${LOG_ROOT}" >&2
  exit "${status}"
fi

sha256sum \
  "${OUTPUT_ROOT}/train/train.jsonl" \
  "${OUTPUT_ROOT}/val/val.jsonl" \
  >"${OUTPUT_ROOT}/sha256.txt"

echo "grounded tool datasets complete: ${OUTPUT_ROOT}"
