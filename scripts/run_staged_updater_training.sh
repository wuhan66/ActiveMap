#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_ROOT/scripts/server_hdpi_env.sh"
# shellcheck source=/dev/null
source "$PROJECT_ROOT/scripts/acquire_training_slot.sh"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/ActiveMap/envs/activemap-gis/bin/python}"
GPU="${ACTIVEMAP_GPU:-0}"
DATA_ROOT="${ACTIVEMAP_STORAGE_ROOT:-${ACTIVEMAP_DATA_ROOT:-/home/wh/ActiveMap}}"
LOG_ROOT="$DATA_ROOT/logs/staged_updater"

if [[ ! "$GPU" =~ ^(0|1)$ ]]; then
  printf 'refusing GPU %s: ActiveMap is restricted to physical GPUs 0 and 1\n' "$GPU" >&2
  exit 2
fi

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

"$PYTHON" scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "$DATA_ROOT"

run_stage() {
  local name="$1"
  local config="$2"
  local run_dir="$3"
  if [[ -s "$run_dir/state.json" ]] && grep -q '"status": "completed"' "$run_dir/state.json"; then
    printf '%s %s already completed; skipping\n' "$(date --iso-8601=seconds)" "$name"
    return
  fi
  printf '%s starting %s on physical GPU %s\n' "$(date --iso-8601=seconds)" "$name" "$GPU"
  "$PYTHON" -m activemap.cli train-updater "$config" 2>&1 | tee "$LOG_ROOT/$name.log"
  printf '%s completed %s\n' "$(date --iso-8601=seconds)" "$name"
}

run_stage \
  inria_segmentation_scratch \
  configs/updater/inria_pretrain_server.yaml \
  "$DATA_ROOT/runs/updater/inria_semantic_segmentation_scratch_seed20260721"

run_stage \
  sn7_vector_change_scratch \
  configs/updater/sn7_v4_hierarchical_vector_change_scratch_server.yaml \
  "$DATA_ROOT/runs/updater/v4_hierarchical_vector_change_scratch_seed20260716"

run_stage \
  muno21_road_scratch \
  configs/updater/muno21_road_v7_topology_scratch_server.yaml \
  "$DATA_ROOT/runs/updater/muno21_road_v7_topology_scratch_seed20260731"
