#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
TRAIN_EPISODES="${TRAIN_EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl}"
ONLINE_REGISTRY="${ONLINE_REGISTRY:-${STORAGE_ROOT}/runs/selector/sn7_mask_features_v2_online_observable_seed20260902_strict_v1/online_controller_registry.yaml}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_recurrent_safe_commit_ab_20260902_v1}"
GPU_IDS=(${GPU_IDS:-0 1 2 3})
ASSET_MAP="${ASSET_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"

common_args=(
  "${TRAIN_EPISODES}" "${ONLINE_REGISTRY}" 20260902
  --storage-root "${STORAGE_ROOT}" --project-root "${PROJECT_ROOT}"
  --device cuda:0 --split train --image-size 512 --budget 3.0
  --max-candidates 16 --max-acquisitions 2 --max-tool-calls 4
  --minimum-chain-length 2 --threshold 0.40 --delta-margin 0
  --min-delta-component-pixels 4 --safe-confidence-threshold 0.68
  --safe-replay-iou-threshold 0.99
  --diagnostic-tool-gate-threshold-override 0.15
  --asset-root-map "${ASSET_MAP}"
  --policy direct_current_hypothesis_safe --policy active_selective_safe
)

run_one() {
  local label="$1"
  local gpu="$2"
  local start="$3"
  local recurrent="$4"
  local recurrent_args=()
  [[ "${recurrent}" == "true" ]] && recurrent_args=(
    --recurrent-safe-commit --retry-confidence-margin 0.05
  )
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/evaluate_online_full_controller.py \
    "${common_args[@]:0:3}" "${RUN_ROOT}/${label}" "${common_args[@]:3}" \
    --chain-start-index "${start}" --max-chains 20 "${recurrent_args[@]}" \
    >"${RUN_ROOT}/logs/${label}.log" 2>&1
}

[[ -x "${PYTHON}" && -f "${TRAIN_EPISODES}" && -f "${ONLINE_REGISTRY}" ]] || {
  echo "missing Python, train episodes, or online registry" >&2
  exit 2
}
mkdir -p "${RUN_ROOT}/logs"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

case "${STAGE}" in
  smoke)
    [[ ! -e "${RUN_ROOT}/smoke" ]] || { echo "smoke output exists" >&2; exit 2; }
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${PYTHON}" \
      scripts/evaluate_online_full_controller.py \
      "${common_args[@]:0:3}" "${RUN_ROOT}/smoke" "${common_args[@]:3}" \
      --chain-index 0 --recurrent-safe-commit --retry-confidence-margin 0.05 \
      >"${RUN_ROOT}/logs/smoke.log" 2>&1
    ;;
  internal_ab)
    [[ -f "${RUN_ROOT}/smoke/summary.json" ]] || {
      echo "internal_ab requires completed smoke" >&2
      exit 2
    }
    [[ ${#GPU_IDS[@]} -ge 4 ]] || { echo "internal_ab requires four GPU ids" >&2; exit 2; }
    labels=(dev_stateless dev_recurrent holdout_stateless holdout_recurrent)
    starts=(0 0 20 20)
    recurrent=(false true false true)
    pids=()
    for index in "${!labels[@]}"; do
      [[ ! -e "${RUN_ROOT}/${labels[$index]}" ]] || {
        echo "output exists: ${RUN_ROOT}/${labels[$index]}" >&2
        exit 2
      }
      run_one "${labels[$index]}" "${GPU_IDS[$index]}" \
        "${starts[$index]}" "${recurrent[$index]}" &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
      wait "${pid}" || status=1
    done
    (( status == 0 )) || exit "${status}"
    ;;
  *)
    echo "usage: $0 smoke|internal_ab" >&2
    exit 2
    ;;
esac

echo "completed ${STAGE} under ${RUN_ROOT}"
