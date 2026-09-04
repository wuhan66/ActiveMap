#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-muno21_sparse_positive_change_seed20260716_v1}"
export POSITIVE_ONLY_CHANGE_DICE=1
export MINIMUM_CALIBRATION_MACRO_F1="${MINIMUM_CALIBRATION_MACRO_F1:-0.38}"

exec bash "$(dirname "$0")/launch_muno21_prior_conditioned_change.sh"
