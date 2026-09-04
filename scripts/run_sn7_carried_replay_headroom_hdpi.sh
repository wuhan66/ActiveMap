#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl}"
BASE_REGISTRY="${BASE_REGISTRY:-${STORAGE_ROOT}/runs/selector/sn7_causal_mask_features_v2_single_seed_20260902_v1/online_controller_registry.yaml}"
CHECKPOINT="${CHECKPOINT:-${STORAGE_ROOT}/runs/updater/sn7_carried_replay_trainval_seed20260902_v2/best_quality.pt}"
REGISTRY="${REGISTRY:-${STORAGE_ROOT}/runs/updater/sn7_carried_replay_trainval_seed20260902_v2/online_candidate_registry.yaml}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_carried_replay_headroom_20260902_v2}"
EVAL_DIR="${EVAL_DIR:-${RUN_ROOT}/eval}"
GPU="${GPU:-3}"
ASSET_MAP="${ASSET_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"

[[ -x "${PYTHON}" && -f "${EPISODES}" && -f "${BASE_REGISTRY}" && -f "${CHECKPOINT}" ]] || {
  echo "missing Python, train-internal episodes, base registry, or updater checkpoint" >&2
  exit 2
}
[[ ! -e "${REGISTRY}" && ! -e "${EVAL_DIR}" ]] || {
  echo "refusing to overwrite registry or evaluation output" >&2
  exit 2
}
mkdir -p "${RUN_ROOT}"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

"${PYTHON}" scripts/derive_online_candidate_registry.py \
  "${BASE_REGISTRY}" "${CHECKPOINT}" "${REGISTRY}" --controller-seed 20260902 \
  --method carried_replay_updater_headroom \
  --purpose train_internal_real_carried_replay_raw_headroom_gate

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
