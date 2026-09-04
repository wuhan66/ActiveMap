#!/usr/bin/env bash
# High-resolution held-out evaluation with a development-frozen novelty threshold.
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 OUTPUT_ROOT SCENE_GLB NOVELTY_THRESHOLD [GPU_IDS]" >&2
  exit 2
fi

output_root=$1
scene=$2
novelty_threshold=$3
gpu_csv=${4:-4,5,6,7}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ACTIVEMAP_AGENT_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}
runner="$repo_root/scripts/run_habitat_online_acquisition_pilot.py"
aggregator="$repo_root/scripts/aggregate_habitat_online_acquisition.py"
seeds=(20270862 20270863 20270864 20270865 20270866 20270867 20270868 20270869)
IFS=',' read -r -a gpu_ids <<<"$gpu_csv"
common_args=(
  --resolution 512
  --grid-resolution-m 0.05
  --ray-stride 2
  --goal-min-distance-m 1
  --goal-max-distance-m 8
)

[[ "$novelty_threshold" =~ ^[0-9]+$ ]] || { echo "Invalid threshold" >&2; exit 2; }
[[ -f "$scene" ]] || { echo "Scene does not exist: $scene" >&2; exit 2; }
[[ ${#gpu_ids[@]} -gt 0 ]] || { echo "At least one GPU ID is required" >&2; exit 2; }
mkdir -p "$output_root"

run_reference() {
  local seed=$1 gpu=$2 output="$output_root/all_seed$1"
  [[ -f "$output/summary.json" && -f "$output/committed_occupancy.npy" ]] && return 0
  [[ ! -e "$output" ]] || { echo "Incomplete output retained: $output" >&2; return 1; }
  "$python_bin" "$runner" "$output" --scene "$scene" --policy acquire_all \
    --gpu-id "$gpu" --seed "$seed" --save-rgb "${common_args[@]}" \
    >"$output_root/all_seed$seed.log" 2>&1
}

run_gate() {
  local policy=$1 label=$2 seed=$3 gpu=$4
  local output="$output_root/${label}_seed$seed"
  local reference="$output_root/all_seed$seed/committed_occupancy.npy"
  [[ -f "$output/summary.json" && -f "$output/committed_occupancy.npy" ]] && return 0
  [[ ! -e "$output" ]] || { echo "Incomplete output retained: $output" >&2; return 1; }
  local threshold_args=(--min-novelty-score "$novelty_threshold")
  if [[ "$policy" == "unknown_gate" ]]; then
    threshold_args=(--min-unknown-score 15)
  fi
  "$python_bin" "$runner" "$output" --scene "$scene" --policy "$policy" \
    "${threshold_args[@]}" --gpu-id "$gpu" --seed "$seed" \
    --reference-occupancy "$reference" "${common_args[@]}" \
    >"$output_root/${label}_seed$seed.log" 2>&1
}

active=0
for index in "${!seeds[@]}"; do
  run_reference "${seeds[$index]}" "${gpu_ids[$((index % ${#gpu_ids[@]}))]}" &
  active=$((active + 1))
  if (( active == ${#gpu_ids[@]} )); then wait; active=0; fi
done
wait

count=0
for seed in "${seeds[@]}"; do
  for spec in "unknown_gate unknown15" "novelty_gate novelty"; do
    read -r policy label <<<"$spec"
    run_gate "$policy" "$label" "$seed" "${gpu_ids[$((count % ${#gpu_ids[@]}))]}" &
    count=$((count + 1))
    if (( count % ${#gpu_ids[@]} == 0 )); then wait; fi
  done
done
wait

expected_count=$(( ${#seeds[@]} * 3 ))
summary_count=$(find "$output_root" -mindepth 2 -maxdepth 2 -name summary.json | wc -l)
[[ "$summary_count" -eq "$expected_count" ]] || {
  echo "Expected $expected_count complete rollouts, found $summary_count" >&2
  exit 1
}
"$python_bin" "$aggregator" "$output_root" "$output_root/aggregate_paired"
