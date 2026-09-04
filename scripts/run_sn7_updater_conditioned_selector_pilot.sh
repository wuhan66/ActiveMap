#!/usr/bin/env bash
# Train and evaluate the pre-registered one-seed C5 selector pilot after audit.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:?set the completed v3 run root}"
F0_GPU="${F0_GPU:-1}"
F1_GPU="${F1_GPU:-3}"

test -s "${RUN_ROOT}/STATE_AUDIT_COMPLETE.json"
for state in f0 f1; do
  for split in train val; do
    test -s "${RUN_ROOT}/states/${state}/${split}/COMPLETE.json"
    test -s "${RUN_ROOT}/states/${state}/${split}/states.jsonl"
  done
done

"${PYTHON}" -c "
import json
from pathlib import Path
summary = json.loads(Path('${RUN_ROOT}/audits/train_step0/summary.json').read_text())
changes = summary['label_transitions']['positive_to_nonpositive'] + summary['label_transitions']['nonpositive_to_positive']
assert changes > 0, 'no train-side label transition; selector pilot is not justified'
"

for gpu in "${F0_GPU}" "${F1_GPU}"; do
  active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${gpu} is occupied: ${active}" >&2; exit 3; }
done

for state in f0 f1; do
  test ! -e "${RUN_ROOT}/selector_data/${state}" || { echo "selector data exists for ${state}" >&2; exit 3; }
  "${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_updater_conditioned_selector_data.py" \
    "${RUN_ROOT}/states/${state}/train/states.jsonl" "${RUN_ROOT}/selector_data/${state}"
done

train() {
  local state="$1" gpu="$2" config="$3"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" -m activemap.cli train-selector "${config}" \
    > "${RUN_ROOT}/selectors/${state}_pilot.log" 2>&1
}
mkdir -p "${RUN_ROOT}/selectors"
train f0 "${F0_GPU}" "${PROJECT_ROOT}/configs/selector/sn7_updater_conditioned_f0_pilot.yaml" &
pid_f0=$!
train f1 "${F1_GPU}" "${PROJECT_ROOT}/configs/selector/sn7_updater_conditioned_f1_pilot.yaml" &
pid_f1=$!
wait "${pid_f0}"
wait "${pid_f1}"

evaluate() {
  local checkpoint="$1" states="$2" output="$3" gpu="$4"
  test ! -e "${output}" || { echo "evaluation output exists: ${output}" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_updater_conditioned_selector.py" \
      "${checkpoint}" "${states}" "${output}" \
      --device cuda:0 --batch-size 128
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
  "${RUN_ROOT}/pilot_assessment" \
  > "${RUN_ROOT}/pilot_assessment.log" 2>&1

printf '{"status":"complete","stage":"one_seed_selector_pilot","test_assets_read":false}\n' \
  > "${RUN_ROOT}/SELECTOR_PILOT_COMPLETE.json"
