#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl}"
REGISTRY="${REGISTRY:-${STORAGE_ROOT}/runs/updater/sn7_carried_prior_residual_trainval_seed20260903/online_candidate_registry.yaml}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_carried_prior_residual_headroom_20260903_v1}"
EVAL_DIR="${EVAL_DIR:-${RUN_ROOT}/eval}"
GPU="${GPU:-3}"
ASSET_MAP="${ASSET_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"

[[ -x "${PYTHON}" && -f "${EPISODES}" && -f "${REGISTRY}" ]] || {
  echo "missing Python, train-internal episodes, or candidate registry" >&2
  exit 2
}
[[ ! -e "${EVAL_DIR}" ]] || {
  echo "refusing to overwrite ${EVAL_DIR}" >&2
  exit 2
}
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_online_full_controller.py \
  "${EPISODES}" "${REGISTRY}" 20260902 "${EVAL_DIR}" \
  --storage-root "${STORAGE_ROOT}" --project-root "${PROJECT_ROOT}" \
  --device cuda:0 --split train --image-size 512 --budget 3.0 \
  --max-candidates 16 --max-acquisitions 2 --max-tool-calls 4 \
  --minimum-chain-length 2 --threshold 0.50 --delta-margin 0.15 \
  --min-delta-component-pixels 4 --safe-confidence-threshold 0.68 \
  --safe-replay-iou-threshold 0.99 --diagnostic-tool-gate-threshold-override 0.15 \
  --causal-evidence-only --recurrent-safe-commit --retry-confidence-margin 0.05 \
  --asset-root-map "${ASSET_MAP}" --policy direct_current_hypothesis_safe \
  --policy active_selective_safe --chain-start-index 0 --max-chains 5 \
  --dump-runtime-oracle-states --runtime-oracle-policy active_selective_safe \
  --runtime-oracle-output-split train --runtime-oracle-utility-profile balanced
