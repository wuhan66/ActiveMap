#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ACTIVEMAP_PYTHON:-/home/wh/venvs/activemap/bin/python}"
GPU="${ACTIVEMAP_GPU:-3}"
DATA_ROOT="${ACTIVEMAP_DATA_ROOT:-/mnt/mydisk/wh/ActiveMap}"
LOG_ROOT="$DATA_ROOT/logs/inria_sn7_transfer"
INRIA_RUN="$DATA_ROOT/runs/updater/inria_segmentation_pretrain_seed20260721"
SN7_RUN="$DATA_ROOT/runs/updater/sn7_vector_change_from_inria_seed20260722"
SN7_SAMPLES="$DATA_ROOT/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl"

mkdir -p "$LOG_ROOT"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$GPU"

is_completed() {
  local run_dir="$1"
  [[ -s "$run_dir/state.json" ]] && grep -q '"status": "completed"' "$run_dir/state.json"
}

run_stage() {
  local name="$1"
  local config="$2"
  local run_dir="$3"
  if is_completed "$run_dir"; then
    echo "[$(date --iso-8601=seconds)] $name already completed; skipping"
    return
  fi
  echo "[$(date --iso-8601=seconds)] starting $name on physical GPU $GPU"
  "$PYTHON" -m activemap.cli train-updater "$config" 2>&1 | tee "$LOG_ROOT/$name.log"
  echo "[$(date --iso-8601=seconds)] completed $name"
}

if [[ ! -s "$DATA_ROOT/processed/inria_v1/updater/APPROVED" ]]; then
  echo "Inria updater data has not received QC approval" >&2
  exit 3
fi

run_stage \
  inria_segmentation_pretrain \
  configs/updater/inria_pretrain_server.yaml \
  "$INRIA_RUN"

run_stage \
  sn7_vector_change_from_inria \
  configs/updater/sn7_from_inria_server.yaml \
  "$SN7_RUN"

if [[ ! -s "$SN7_RUN/promotion_decision.json" ]]; then
  "$PYTHON" scripts/finalize_updater_hierarchy.py \
    "$SN7_RUN" "$SN7_SAMPLES" \
    --device cuda --batch-size 64 --validation-only \
    2>&1 | tee "$LOG_ROOT/sn7_validation_finalize.log"
fi
