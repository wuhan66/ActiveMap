#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl}"
REGISTRY="${REGISTRY:-${STORAGE_ROOT}/runs/selector/sn7_causal_mask_features_v2_single_seed_20260902_v1/online_controller_registry.yaml}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_carried_runtime_oracle_20260902_v1}"
GPU_DEV="${GPU_DEV:-2}"
GPU_HOLDOUT="${GPU_HOLDOUT:-3}"
ASSET_MAP="${ASSET_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"

[[ -x "${PYTHON}" && -f "${EPISODES}" && -f "${REGISTRY}" ]] || {
  echo "missing Python, episodes, or controller registry" >&2
  exit 2
}
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "refusing to overwrite ${RUN_ROOT}" >&2
  exit 2
}
mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

run_partition() {
  local label="$1"
  local gpu="$2"
  local start="$3"
  local output_split="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/evaluate_online_full_controller.py \
    "${EPISODES}" "${REGISTRY}" 20260902 "${RUN_ROOT}/${label}" \
    --storage-root "${STORAGE_ROOT}" --project-root "${PROJECT_ROOT}" \
    --device cuda:0 --split train --image-size 512 --budget 3.0 \
    --max-candidates 16 --max-acquisitions 2 --max-tool-calls 4 \
    --minimum-chain-length 2 --threshold 0.50 --delta-margin 0.15 \
    --min-delta-component-pixels 4 --safe-confidence-threshold 0.68 \
    --safe-replay-iou-threshold 0.99 \
    --diagnostic-tool-gate-threshold-override 0.15 \
    --causal-evidence-only --recurrent-safe-commit \
    --retry-confidence-margin 0.05 --asset-root-map "${ASSET_MAP}" \
    --policy direct_current_hypothesis_safe \
    --policy active_selective_safe \
    --chain-start-index "${start}" --max-chains 20 \
    --dump-runtime-oracle-states \
    --runtime-oracle-policy active_selective_safe \
    --runtime-oracle-output-split "${output_split}" \
    --runtime-oracle-utility-profile balanced \
    >"${RUN_ROOT}/logs/${label}.log" 2>&1
}

run_partition dev "${GPU_DEV}" 0 train &
dev_pid=$!
run_partition holdout "${GPU_HOLDOUT}" 20 val &
holdout_pid=$!
status=0
wait "${dev_pid}" || status=1
wait "${holdout_pid}" || status=1
(( status == 0 )) || exit "${status}"

"${PYTHON}" scripts/merge_selector_state_files.py \
  "${RUN_ROOT}/states_train_val_carried_runtime_oracle.jsonl" \
  "${RUN_ROOT}/dev/runtime_oracle_states.jsonl" \
  "${RUN_ROOT}/holdout/runtime_oracle_states.jsonl" \
  >"${RUN_ROOT}/logs/merge.log" 2>&1

echo "completed carried-state runtime oracle under ${RUN_ROOT}"
