#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/argotweak-legacy/bin/python"
OFFICIAL="${STORE}/external/ArgoTweak_baselines"
CONFIG="${OFFICIAL}/projects/configs/argotweak_explainable.py"
ANN="${STORE}/datasets/argotweak/tbv_balanced_24_8_v1/official_full/val_argotweak_balanced8.pkl"
SEED1="${STORE}/runs/argotweak/full_domain_adaptation_balanced24_v1/four_gpu_seed20260831/epoch_10.pth"
ROOT="${STORE}/runs/argotweak/baseline_vs_adapted_epoch10_v2"
LOG_ROOT="${STORE}/logs/argotweak_baseline_vs_adapted_epoch10_v2"

mkdir -p "${ROOT}" "${LOG_ROOT}"
exec 9>"${ROOT}/.launcher.lock"
flock -n 9 || exit 0
[[ ! -e "${ROOT}/MATRIX_COMPLETED" ]] || exit 0
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

run_one() {
  local label="$1" checkpoint="$2"
  local output="${ROOT}/${label}"
  local log="${LOG_ROOT}/${label}.log"
  mkdir -p "${output}"
  if [[ ! -s "${output}/results.pkl" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u scripts/run_argotweak_official_test.py \
      --official-root "${OFFICIAL}" --config "${CONFIG}" --annotations "${ANN}" \
      --checkpoint "${checkpoint}" --output-dir "${output}" --workers 4 \
      >"${log}" 2>&1
  fi
  if [[ ! -s "${output}/atomic_edit_proposals.jsonl" ]]; then
    "${PYTHON}" scripts/export_argotweak_official_proposals.py \
      --results "${output}/results.pkl" --annotations "${ANN}" \
      --output "${output}/atomic_edit_proposals.jsonl" --object-threshold 0.3 \
      >>"${log}" 2>&1
  fi
}

status=0
run_one official_baseline "${OFFICIAL}/checkpoints/argotweak_baseline.pth" || status=1
run_one adapted_epoch10 "${SEED1}" || status=1
if [[ "${status}" -eq 0 ]]; then
  touch "${ROOT}/MATRIX_COMPLETED"
else
  touch "${ROOT}/MATRIX_FAILED"
fi
exit "${status}"
