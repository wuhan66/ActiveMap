#!/usr/bin/env bash
set -euo pipefail

STORAGE_ROOT="${STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
export RUN_NAME="${RUN_NAME:-muno21_prior_conditioned_operation_refine_seed20260716_v1}"
export OPERATION_ONLY_FROM="${OPERATION_ONLY_FROM:-${STORAGE_ROOT}/runs/semantic/muno21_prior_conditioned_change_seed20260716_v2/checkpoints/best_head.pt}"
export MINIMUM_CALIBRATION_MACRO_F1="${MINIMUM_CALIBRATION_MACRO_F1:-0.38}"
export MAX_EPOCHS="${MAX_EPOCHS:-40}"
export PATIENCE="${PATIENCE:-8}"
export LEARNING_RATE="${LEARNING_RATE:-1e-3}"

exec bash "$(dirname "$0")/launch_muno21_prior_conditioned_change.sh"
