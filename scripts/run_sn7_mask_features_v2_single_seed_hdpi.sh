#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"
STORE="${ACTIVEMAP_STORE:-/home/wh/ActiveMap}"
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
STATES="${STATES:-${STORE}/runs/selector/sn7_mask_features_v2_full_20260901/states_train_val_budget3_runtime_grid_mask_v2.jsonl}"
OUTPUT="${OUTPUT:-${STORE}/runs/selector/sn7_mask_features_v2_single_seed_20260902_v2}"
GPU="${GPU:-2}"
MAX_FALSE_CALL="${MAX_FALSE_CALL:-0.02}"
MAX_HARMFUL_CALLS="${MAX_HARMFUL_CALLS:-0.20}"
MIN_ACQUIRE_RECALL="${MIN_ACQUIRE_RECALL:-0.10}"

export PYTHONPATH="${REPO}:${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO}"

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite ${OUTPUT}" >&2
  exit 2
fi
if [[ ! -s "${STATES}" ]]; then
  echo "missing immutable train/validation states: ${STATES}" >&2
  exit 3
fi

mkdir -p "${OUTPUT}"
CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  -m scripts.train_topk_evidence_reranker \
  "${STATES}" \
  "${OUTPUT}/train" \
  --device cuda \
  --seed 20260902 \
  --feature-set policy-relative-mask \
  --top-k 3 \
  --calibration-fraction 0.2 \
  --calibration-group-key aoi_id \
  --epochs 50 \
  --warmup-epochs 5 \
  --batch-size 128 \
  --pairwise-weight 2.0 \
  --utility-positive-weight 16 \
  --beneficial-positive-weight 16 \
  --maximum-false-call-rate "${MAX_FALSE_CALL}" \
  --maximum-harmful-call-fraction "${MAX_HARMFUL_CALLS}" \
  --minimum-acquire-recall "${MIN_ACQUIRE_RECALL}" \
  >"${OUTPUT}/train.log" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" \
  -m scripts.audit_sn7_v6_evidence_value_policy \
  "${OUTPUT}/train/best.pt" \
  "${STATES}" \
  "${OUTPUT}/validation" \
  --split val \
  --device cuda \
  >"${OUTPUT}/validation.log" 2>&1

echo "SN7 mask-v2 single-seed validation complete: ${OUTPUT}"
