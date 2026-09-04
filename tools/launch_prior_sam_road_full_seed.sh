#!/usr/bin/env bash
set -euo pipefail

SEED="${PRIOR_SAM_ROAD_CHANGE_SEED:?set PRIOR_SAM_ROAD_CHANGE_SEED}"
GPU="${PRIOR_SAM_ROAD_GPU:?set PRIOR_SAM_ROAD_GPU}"

[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "invalid seed: $SEED" >&2; exit 2; }
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "invalid GPU: $GPU" >&2; exit 2; }

export PRIOR_SAM_ROAD_RUN_NAME="${PRIOR_SAM_ROAD_RUN_NAME:-post_acquisition_prior_sam_road_aligned_full_seed${SEED}_v1}"
export PRIOR_SAM_ROAD_TRAIN_LIMIT=all
export PRIOR_SAM_ROAD_VAL_LIMIT=all
export PRIOR_SAM_ROAD_EVAL_SEED="${PRIOR_SAM_ROAD_EVAL_SEED:-$SEED}"
export PRIOR_SAM_ROAD_RUN_EVAL=1

exec bash "$(dirname "$0")/launch_prior_sam_road_belief_gate.sh"
