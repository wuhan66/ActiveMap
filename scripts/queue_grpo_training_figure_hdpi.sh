#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
MATRIX_ROOT="${1:-/home/wh/ActiveMap/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n512_matrix_v1}"
OUTPUT_ROOT="${2:-/home/wh/ActiveMap/reports/muno21_grpo_balanced_validation_20260802/training_dynamics}"
EXPECTED_RUNS="${EXPECTED_RUNS:-8}"
PYTHON="${PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}"

while [[ $(find "${MATRIX_ROOT}" -mindepth 2 -maxdepth 2 -name COMPLETED -type f | wc -l) -lt "${EXPECTED_RUNS}" ]]; do
  sleep 60
done

cd "${PROJECT_ROOT}"
PYTHONPATH=src:. "${PYTHON}" scripts/plot_grpo_training_dynamics.py \
  "${MATRIX_ROOT}" "${OUTPUT_ROOT}"
date -Is >"${OUTPUT_ROOT}/COMPLETED"
