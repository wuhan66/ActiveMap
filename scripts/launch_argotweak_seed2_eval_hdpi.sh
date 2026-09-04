#!/usr/bin/env bash
set -euo pipefail
PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/argotweak-legacy/bin/python"
OFFICIAL="${STORE}/external/ArgoTweak_baselines"
CHECKPOINT="${STORE}/runs/argotweak/full_domain_adaptation_balanced24_v1/four_gpu_seed20260832/epoch_10.pth"
ANN="${STORE}/datasets/argotweak/tbv_balanced_24_8_v1/official_full/val_argotweak_balanced8.pkl"
ROOT="${STORE}/runs/argotweak/seed2_epoch10_official_eval_v1"
mkdir -p "${ROOT}"
exec 9>"${ROOT}/.lock"; flock -n 9 || exit 0
[[ ! -e "${ROOT}/MATRIX_COMPLETED" ]] || exit 0
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}/src:${PROJECT}${PYTHONPATH:+:${PYTHONPATH}}"
CUDA_VISIBLE_DEVICES="${GPU:-0}" "${PYTHON}" -u scripts/run_argotweak_official_test.py \
  --official-root "${OFFICIAL}" --config "${OFFICIAL}/projects/configs/argotweak_explainable.py" \
  --annotations "${ANN}" --checkpoint "${CHECKPOINT}" --output-dir "${ROOT}/evaluation" \
  --workers 4 --seed 20260832 >"${ROOT}/evaluation.log" 2>&1
"${PYTHON}" scripts/export_argotweak_official_proposals.py \
  --results "${ROOT}/evaluation/results.pkl" --annotations "${ANN}" \
  --output "${ROOT}/atomic_edit_proposals.jsonl" --object-threshold 0.3 \
  --object-match-distance 1.5 >>"${ROOT}/evaluation.log" 2>&1
touch "${ROOT}/MATRIX_COMPLETED"
