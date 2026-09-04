#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"

declare -a JOBS=(
  "0|sn7_v4_hierarchical_vector_change_scratch_seed4_server.yaml|v4_hierarchical_vector_change_scratch_seed20260719"
  "1|sn7_v4_hierarchical_vector_change_scratch_seed5_server.yaml|v4_hierarchical_vector_change_scratch_seed20260722"
)

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "$STORAGE_ROOT"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r gpu config_name run_name <<<"$spec"
  config="$PROJECT_ROOT/configs/updater/$config_name"
  run_dir="$STORAGE_ROOT/runs/updater/$run_name"
  archive_dir="$PROJECT_ROOT/outputs/updater/$run_name"

  gpu_pids="$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)"
  if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
    echo "Refusing GPU $gpu: active compute PIDs: $gpu_pids" >&2
    exit 2
  fi
  if [[ -e "$run_dir/train.pid" ]] || [[ -e "$run_dir/best_quality.pt" ]]; then
    echo "Refusing existing run state: $run_dir" >&2
    exit 3
  fi
  mkdir -p "$run_dir"
  printf 'CUDA_VISIBLE_DEVICES=%q %q -m activemap.cli train-updater %q --output %q\n' \
    "$gpu" "$PYTHON" "$config" "$run_dir" >"$run_dir/launch_command.txt"

  nohup env CUDA_VISIBLE_DEVICES="$gpu" ACTIVEMAP_PYTHON="$PYTHON" \
    UPDATER_PHYSICAL_GPU="$gpu" UPDATER_CONFIG="$config" \
    UPDATER_RUN_DIR="$run_dir" UPDATER_ARCHIVE_DIR="$archive_dir" \
    bash "$PROJECT_ROOT/scripts/run_updater_training_job.sh" \
    >"$run_dir/train.log" 2>&1 < /dev/null &
  pid=$!
  echo "$pid" >"$run_dir/train.pid"
  echo "started seed run=$run_name gpu=$gpu pid=$pid"
done

sleep 5
for spec in "${JOBS[@]}"; do
  IFS='|' read -r gpu _ run_name <<<"$spec"
  run_dir="$STORAGE_ROOT/runs/updater/$run_name"
  pid="$(cat "$run_dir/train.pid")"
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Launch failed for $run_name; inspect $run_dir/train.log" >&2
    exit 4
  fi
  echo "healthy seed run=$run_name gpu=$gpu pid=$pid"
done
