#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
RUN_ROOT="${TOOL_BELIEF_RUN_ROOT:-/mnt/mydisk/wh/ActiveMap/runs/agent}"
GPU="${GPU:-3}"
SEEDS=(20260901 20260902 20260903)
OUTPUT_DIR="${RUN_ROOT}/tool_belief_spatial_v2_3seed"
LOG_ROOT="${LOG_ROOT:-/mnt/mydisk/wh/ActiveMap/logs}"
STATUS_PATH="${LOG_ROOT}/tool_belief_spatial_v2_3seed_master.log.exit_code"

mkdir -p "$LOG_ROOT"
rm -f "$STATUS_PATH"
trap 'status=$?; printf "%s\n" "$status" > "$STATUS_PATH"' EXIT

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

for seed in "${SEEDS[@]}"; do
  echo "Starting gated-residual spatial/no-spatial pair for seed ${seed}"
  GPU="$GPU" EXPERIMENT_VERSION=v2 FUSION_MODE=gated_residual GATE_BIAS=-2.0 \
    bash scripts/run_tool_belief_spatial_seed_pair.sh "$seed"
done

aggregate_args=()
for seed in "${SEEDS[@]}"; do
  spatial="${RUN_ROOT}/tool_belief_spatial_decision_v2_spatial_seed${seed}_eval/summary.json"
  no_spatial="${RUN_ROOT}/tool_belief_spatial_decision_v2_no_spatial_seed${seed}_eval/summary.json"
  aggregate_args+=(--pair "${seed}=${spatial},${no_spatial}")
done

"$PYTHON" scripts/aggregate_tool_belief_spatial_seeds.py \
  "${aggregate_args[@]}" --output-dir "$OUTPUT_DIR" \
  --minimum-mean-gain 0.01 --safety-margin 0.02

echo "Completed gated-residual spatial v2 three-seed protocol"
