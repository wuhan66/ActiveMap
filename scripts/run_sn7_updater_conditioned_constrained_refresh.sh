#!/usr/bin/env bash
# Evaluate a pre-calibrated false-edit guard for the C5 refresh mechanism.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:?set the completed C5 v3 run root}"
STALE_GPU="${STALE_GPU:-1}"
REFRESHED_GPU="${REFRESHED_GPU:-3}"
MAX_FALSE_EDIT_INCREASE="${MAX_FALSE_EDIT_INCREASE:-0.0}"
GRID_STEPS="${GRID_STEPS:-21}"

test -s "${RUN_ROOT}/STATE_AUDIT_COMPLETE.json"
for state in f0 f1; do
  test -s "${RUN_ROOT}/selector_data/${state}/selector_data.jsonl"
  test -s "${RUN_ROOT}/selectors/${state}_pilot/best.pt"
done
test -s "${RUN_ROOT}/states/f1/val/states.jsonl"

OUT="${RUN_ROOT}/constrained_refresh"
test ! -e "${OUT}" || { echo "refusing to overwrite ${OUT}" >&2; exit 3; }
for gpu in "${STALE_GPU}" "${REFRESHED_GPU}"; do
  active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${gpu} is occupied: ${active}" >&2; exit 3; }
done
mkdir -p "${OUT}"

# This reads only the f1 train/dev partition created from the f1 train cache.
CUDA_VISIBLE_DEVICES="${REFRESHED_GPU}" PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/calibrate_updater_conditioned_risk_cap.py" \
  "${RUN_ROOT}/selectors/f0_pilot/best.pt" \
  "${RUN_ROOT}/selectors/f1_pilot/best.pt" \
  "${RUN_ROOT}/selector_data/f1/selector_data.jsonl" \
  "${OUT}/train_dev_calibration" \
  --device cuda:0 --batch-size 128 --grid-steps "${GRID_STEPS}" \
  --max-false-edit-increase "${MAX_FALSE_EDIT_INCREASE}" \
  > "${OUT}/train_dev_calibration.log" 2>&1

RISK_CAP="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["risk_cap"])' \
  "${OUT}/train_dev_calibration/risk_cap_calibration.json")"
printf '{"risk_cap":%s,"selection_split":"f1_train_dev","test_assets_read":false}\n' "${RISK_CAP}" \
  > "${OUT}/FROZEN_RISK_CAP.json"

evaluate() {
  local checkpoint="$1" output="$2" gpu="$3" risk_cap="$4"
  local args=("${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_updater_conditioned_selector.py"
    "${checkpoint}" "${RUN_ROOT}/states/f1/val/states.jsonl" "${output}"
    --device cuda:0 --batch-size 128 --split val)
  if [[ "${risk_cap}" != "none" ]]; then
    args+=(--max-candidate-risk "${risk_cap}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" "${args[@]}"
}

evaluate "${RUN_ROOT}/selectors/f0_pilot/best.pt" "${OUT}/f0_on_f1_stale" "${STALE_GPU}" none &
pid_stale=$!
evaluate "${RUN_ROOT}/selectors/f1_pilot/best.pt" "${OUT}/f1_on_f1_constrained" "${REFRESHED_GPU}" "${RISK_CAP}" &
pid_refreshed=$!
wait "${pid_stale}"
wait "${pid_refreshed}"

"${PYTHON}" "${PROJECT_ROOT}/scripts/assess_updater_conditioned_selector_pilot.py" \
  "${OUT}/f0_on_f1_stale/per_sample.jsonl" \
  "${OUT}/f0_on_f1_stale/per_sample.jsonl" \
  "${OUT}/f1_on_f1_constrained/per_sample.jsonl" \
  "${OUT}/assessment" \
  > "${OUT}/assessment.log" 2>&1

printf '{"status":"complete","stage":"constrained_updater_refresh","risk_cap":%s,"test_assets_read":false}\n' "${RISK_CAP}" \
  > "${OUT}/CONSTRAINED_REFRESH_COMPLETE.json"
