#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
SWEEP_ROOT="${SWEEP_ROOT:-${STORAGE_ROOT}/runs/selector/sn7_executable_two_stage_sweep_seed20260721_v1}"
THREE_SEED_ROOT="${THREE_SEED_ROOT:-${STORAGE_ROOT}/runs/selector/sn7_executable_two_stage_three_seed_v1}"
SAMPLES="${SAMPLES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v2/states_train_val_executable_balanced_m15.jsonl}"
DECISION="${SWEEP_ROOT}/promotion_decision.json"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"

while [[ ! -s "${SWEEP_ROOT}/process_results.json" ]]; do
  sleep "${POLL_SECONDS}"
done

"${PYTHON}" scripts/finalize_sn7_executable_selector_sweep.py \
  "${SWEEP_ROOT}" --output "${DECISION}"

VARIANT="$(${PYTHON} -c '
import json, sys
decision = json.load(open(sys.argv[1], encoding="utf-8"))
if not decision["promoted"]:
    raise SystemExit(3)
print(decision["winner"]["run"].rsplit("_seed", 1)[0])
' "${DECISION}")"

[[ ! -e "${THREE_SEED_ROOT}" ]] || {
  echo "Refusing to overwrite ${THREE_SEED_ROOT}" >&2
  exit 5
}

"${PYTHON}" scripts/launch_sn7_two_stage_three_seed.py \
  "${THREE_SEED_ROOT}" \
  --gpu 0 --gpu 2 --gpu 4 \
  --seed 20260721 --seed 20260722 --seed 20260723 \
  --variant "${VARIANT}" --samples "${SAMPLES}" --python "${PYTHON}"
