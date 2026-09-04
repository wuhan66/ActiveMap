#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
SAMPLES="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl"
EVAL_ROOT="${STORAGE_ROOT}/runs/updater/concat_unet_three_seed_val_20260729"
LOG="${STORAGE_ROOT}/logs/concat_unet_three_seed_completion_20260729.log"

declare -a JOBS=(
  "0|20260719|v4_hierarchical_vector_change_scratch_seed20260719"
  "1|20260722|v4_hierarchical_vector_change_scratch_seed20260722"
)

mkdir -p "$(dirname "$LOG")" "$EVAL_ROOT"
exec >>"$LOG" 2>&1
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[$(date --iso-8601=seconds)] waiting for concat U-Net seed completion"

for spec in "${JOBS[@]}"; do
  IFS='|' read -r _ seed run_name <<<"$spec"
  run_dir="$STORAGE_ROOT/runs/updater/$run_name"
  while true; do
    if grep -q '"status": "completed"' "$run_dir/state.json" 2>/dev/null; then
      break
    fi
    if [[ -s "$run_dir/train.pid" ]]; then
      pid="$(cat "$run_dir/train.pid")"
      if ! kill -0 "$pid" 2>/dev/null; then
        echo "training failed before completion: seed=$seed run=$run_dir" >&2
        exit 5
      fi
    fi
    sleep 120
  done
done

declare -a EVAL_PIDS=()
for spec in "${JOBS[@]}"; do
  IFS='|' read -r gpu seed run_name <<<"$spec"
  run_dir="$STORAGE_ROOT/runs/updater/$run_name"
  output_dir="$EVAL_ROOT/seed$seed"
  if [[ -s "$output_dir/summary.json" ]]; then
    echo "seed $seed evaluation already complete"
    continue
  fi
  mkdir -p "$output_dir"
  if [[ -s "$output_dir/eval.pid" ]]; then
    existing_pid="$(cat "$output_dir/eval.pid")"
    existing_cmd="$(ps -p "$existing_pid" -o cmd= 2>/dev/null || true)"
    if [[ "$existing_cmd" == *"evaluate-updater"*"$output_dir"* ]]; then
      echo "waiting for existing validation evaluation seed=$seed pid=$existing_pid"
      while kill -0 "$existing_pid" 2>/dev/null; do
        sleep 30
      done
      if [[ -s "$output_dir/summary.json" ]]; then
        echo "existing validation evaluation completed seed=$seed"
        continue
      fi
      echo "existing validation evaluation failed seed=$seed" >&2
      exit 6
    fi
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -m activemap.cli evaluate-updater \
    "$run_dir/best_quality.pt" "$SAMPLES" "$output_dir" \
    --split val --device auto --batch-size 64 --num-workers 4 \
    --bootstrap 5000 --seed "$seed" --edit-decoding auto \
    >"$output_dir/eval.log" 2>&1 &
  eval_pid="$!"
  printf '%s\n' "$eval_pid" >"$output_dir/eval.pid"
  EVAL_PIDS+=("$eval_pid")
  echo "started validation evaluation seed=$seed gpu=$gpu pid=$eval_pid"
done
for pid in "${EVAL_PIDS[@]}"; do
  wait "$pid"
done

"$PYTHON" scripts/aggregate_sn7_concat_unet_validation.py \
  "$EVAL_ROOT/paper_artifacts" \
  --summary "20260716=$EVAL_ROOT/seed20260716/summary.json" \
  --summary "20260719=$EVAL_ROOT/seed20260719/summary.json" \
  --summary "20260722=$EVAL_ROOT/seed20260722/summary.json"

"$PYTHON" scripts/plot_sn7_concat_unet_training.py \
  "$EVAL_ROOT/training_curves" \
  --history "seed20260719=$STORAGE_ROOT/runs/updater/v4_hierarchical_vector_change_scratch_seed20260719/history.jsonl" \
  --history "seed20260722=$STORAGE_ROOT/runs/updater/v4_hierarchical_vector_change_scratch_seed20260722/history.jsonl"

echo "[$(date --iso-8601=seconds)] concat U-Net three-seed validation complete"
