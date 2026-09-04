#!/usr/bin/env bash
set -euo pipefail

export RUN_NAME="${RUN_NAME:-muno21_operation_spatial_refine_seed20260716_v1}"
export OPERATION_HEAD="spatial_pyramid"
export MAX_EPOCHS="${MAX_EPOCHS:-40}"
export PATIENCE="${PATIENCE:-8}"
export LEARNING_RATE="${LEARNING_RATE:-5e-4}"

exec bash "$(dirname "$0")/launch_muno21_operation_refine.sh"
