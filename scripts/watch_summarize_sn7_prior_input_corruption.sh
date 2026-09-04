#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba/prior_input_corruption_v1"
POLL_SECONDS="${POLL_SECONDS:-60}"

for severity in 0 4 8 16; do
  for seed in 20260725 20260726 20260727; do
    summary="${ROOT}/severity${severity}_model_seed${seed}/summary.json"
    until [[ -s "${summary}" ]]; do
      sleep "${POLL_SECONDS}"
    done
  done
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" scripts/summarize_sn7_prior_input_corruption.py "${ROOT}" \
  --bootstrap-repetitions 5000 --bootstrap-seed 20260729 \
  > "${ROOT}/summarizer.log" 2>&1
"${PYTHON}" scripts/plot_sn7_prior_input_corruption.py \
  "${ROOT}/summary.json" "${ROOT}/prior_input_corruption_curve"
