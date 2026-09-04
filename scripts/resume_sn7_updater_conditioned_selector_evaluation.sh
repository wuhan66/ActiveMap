#!/usr/bin/env bash
# Resume only the immutable C5 validation evaluation after an evaluator repair.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:?set the completed C5 run root}"
F0_GPU="${F0_GPU:-3}"
F1_GPU="${F1_GPU:-5}"

test -s "${RUN_ROOT}/STATE_AUDIT_COMPLETE.json"
test -s "${RUN_ROOT}/selectors/f0_pilot/best.pt"
test -s "${RUN_ROOT}/selectors/f1_pilot/best.pt"
test ! -e "${RUN_ROOT}/SELECTOR_PILOT_COMPLETE.json" || {
  echo "pilot already completed" >&2
  exit 3
}

evaluate() {
  local checkpoint="$1" states="$2" output="$3" gpu="$4"
  test ! -e "${output}" || { echo "evaluation output exists: ${output}" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_updater_conditioned_selector.py" \
      "${checkpoint}" "${states}" "${output}" --device cuda:0 --batch-size 128
}

evaluate "${RUN_ROOT}/selectors/f0_pilot/best.pt" "${RUN_ROOT}/states/f0/val/states.jsonl" \
  "${RUN_ROOT}/evaluation/f0_on_f0" "${F0_GPU}" &
pid_matched=$!
evaluate "${RUN_ROOT}/selectors/f1_pilot/best.pt" "${RUN_ROOT}/states/f1/val/states.jsonl" \
  "${RUN_ROOT}/evaluation/f1_on_f1" "${F1_GPU}" &
pid_refreshed=$!
wait "${pid_matched}"
wait "${pid_refreshed}"

evaluate "${RUN_ROOT}/selectors/f0_pilot/best.pt" "${RUN_ROOT}/states/f1/val/states.jsonl" \
  "${RUN_ROOT}/evaluation/f0_on_f1" "${F0_GPU}"

test ! -e "${RUN_ROOT}/pilot_assessment" || { echo "pilot assessment output exists" >&2; exit 3; }
"${PYTHON}" "${PROJECT_ROOT}/scripts/assess_updater_conditioned_selector_pilot.py" \
  "${RUN_ROOT}/evaluation/f0_on_f0/per_sample.jsonl" \
  "${RUN_ROOT}/evaluation/f0_on_f1/per_sample.jsonl" \
  "${RUN_ROOT}/evaluation/f1_on_f1/per_sample.jsonl" \
  "${RUN_ROOT}/pilot_assessment"

printf '{"status":"complete","stage":"one_seed_selector_pilot","resumed_evaluation":true,"test_assets_read":false}\n' \
  > "${RUN_ROOT}/SELECTOR_PILOT_COMPLETE.json"
