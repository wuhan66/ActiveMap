#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl}"
CANONICAL="${CANONICAL:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1/updater_samples.jsonl}"
DATA_ROOT="${DATA_ROOT:-${STORAGE_ROOT}/processed/sn7_v1/updater_carried_runtime_temporal_20260903_v2}"
REPLAY="${REPLAY:-${DATA_ROOT}/updater_samples.jsonl}"
MERGED="${MERGED:-${DATA_ROOT}/updater_samples_canonical_replay_v2.jsonl}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/configs/updater/sn7_carried_runtime_temporal_seed20260903_server.yaml}"
TRAIN_ROOT="${TRAIN_ROOT:-${STORAGE_ROOT}/runs/updater/sn7_carried_runtime_temporal_seed20260903}"
CHECKPOINT="${CHECKPOINT:-${TRAIN_ROOT}/best_quality.pt}"
BASE_REGISTRY="${BASE_REGISTRY:-${STORAGE_ROOT}/runs/selector/sn7_mask_features_v2_online_observable_seed20260902_strict_v1/online_controller_registry.yaml}"
REGISTRY="${REGISTRY:-${TRAIN_ROOT}/online_candidate_registry.yaml}"
GATE_ROOT="${GATE_ROOT:-${STORAGE_ROOT}/runs/sn7_carried_runtime_temporal_headroom_20260903_v1}"
GPU="${GPU:-3}"
CONTROLLER_SEED="${CONTROLLER_SEED:-20260902}"
ASSET_MAP="${ASSET_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"

[[ -x "${PYTHON}" && -f "${EPISODES}" && -f "${CANONICAL}" && -f "${CONFIG}" ]] || {
  echo "missing Python, train-internal episodes, temporal manifest, or config" >&2
  exit 2
}
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

if [[ ! -f "${REPLAY}" ]]; then
  "${PYTHON}" scripts/build_sn7_carried_runtime_temporal_dataset.py \
    "${EPISODES}" "${DATA_ROOT}" --image-size 512 \
    --minimum-chain-length 2 --validation-chain-index 20 \
    --max-chains 40 \
    --asset-root-map "${ASSET_MAP}"
fi
if [[ ! -f "${MERGED}" ]]; then
  "${PYTHON}" scripts/prepare_sn7_carried_runtime_temporal_manifest.py \
    "${CANONICAL}" "${REPLAY}" "${MERGED}"
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -m activemap.cli train-updater "${CONFIG}"
fi

if [[ ! -f "${REGISTRY}" ]]; then
  "${PYTHON}" scripts/derive_online_candidate_registry.py \
    "${BASE_REGISTRY}" "${CHECKPOINT}" "${REGISTRY}" \
    --controller-seed "${CONTROLLER_SEED}" \
    --method carried_runtime_temporal_updater_headroom \
    --purpose train_internal_aligned_temporal_raw_headroom_gate
fi
[[ ! -e "${GATE_ROOT}/eval" ]] || {
  echo "refusing to overwrite gate output: ${GATE_ROOT}/eval" >&2
  exit 2
}
mkdir -p "${GATE_ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_online_full_controller.py \
  "${EPISODES}" "${REGISTRY}" "${CONTROLLER_SEED}" "${GATE_ROOT}/eval" \
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
