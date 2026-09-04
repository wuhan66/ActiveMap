#!/usr/bin/env bash
# Held-out matched-path evaluation after novelty threshold selection on development seed.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 OUTPUT_ROOT SCENE_GLB [GPU_IDS]" >&2
  exit 2
fi

output_root=$1
scene=$2
gpu_csv=${3:-0,1,2,3,4,5,6,7}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ACTIVEMAP_AGENT_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}
runner="$repo_root/scripts/run_habitat_online_acquisition_pilot.py"
aggregator="$repo_root/scripts/aggregate_habitat_online_acquisition.py"
seeds=(20270862 20270863 20270864 20270865 20270866 20270867 20270868 20270869)
IFS=',' read -r -a gpu_ids <<<"$gpu_csv"

[[ -f "$scene" ]] || { echo "Scene does not exist: $scene" >&2; exit 2; }
[[ ${#gpu_ids[@]} -gt 0 ]] || { echo "At least one GPU ID is required" >&2; exit 2; }
for index in "${!gpu_ids[@]}"; do
  gpu=${gpu_ids[$index]}
  [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU ID: $gpu" >&2; exit 2; }
  for prior in "${gpu_ids[@]:0:index}"; do
    [[ "$gpu" != "$prior" ]] || { echo "Duplicate GPU ID: $gpu" >&2; exit 2; }
  done
done
mkdir -p "$output_root"

run_reference() {
  local seed=$1
  local gpu=$2
  local output="$output_root/all_seed$seed"
  [[ -f "$output/summary.json" && -f "$output/committed_occupancy.npy" ]] && return 0
  [[ ! -e "$output" ]] || {
    echo "Incomplete rollout directory retained for audit: $output" >&2
    return 1
  }
  "$python_bin" "$runner" "$output" --scene "$scene" --policy acquire_all \
    --gpu-id "$gpu" --seed "$seed" --goal-min-distance-m 1 --goal-max-distance-m 8 \
    >"$output_root/all_seed$seed.log" 2>&1
}

run_gate() {
  local policy=$1
  local label=$2
  local seed=$3
  local gpu=$4
  local output="$output_root/${label}_seed$seed"
  local reference="$output_root/all_seed$seed/committed_occupancy.npy"
  [[ -f "$output/summary.json" && -f "$output/committed_occupancy.npy" ]] && return 0
  [[ ! -e "$output" ]] || {
    echo "Incomplete rollout directory retained for audit: $output" >&2
    return 1
  }
  if [[ "$policy" == "unknown_gate" ]]; then
    "$python_bin" "$runner" "$output" --scene "$scene" --policy unknown_gate \
      --min-unknown-score 15 --gpu-id "$gpu" --seed "$seed" \
      --reference-occupancy "$reference" --goal-min-distance-m 1 --goal-max-distance-m 8 \
      >"$output_root/${label}_seed$seed.log" 2>&1
  else
    "$python_bin" "$runner" "$output" --scene "$scene" --policy novelty_gate \
      --min-novelty-score 8 --gpu-id "$gpu" --seed "$seed" \
      --reference-occupancy "$reference" --goal-min-distance-m 1 --goal-max-distance-m 8 \
      >"$output_root/${label}_seed$seed.log" 2>&1
  fi
}

active=0
for index in "${!seeds[@]}"; do
  run_reference "${seeds[$index]}" "${gpu_ids[$((index % ${#gpu_ids[@]}))]}" &
  active=$((active + 1))
  if (( active == ${#gpu_ids[@]} )); then
    wait
    active=0
  fi
done
wait
reference_count=$(find "$output_root" -maxdepth 2 -path '*/all_seed*/summary.json' | wc -l)
if [[ "$reference_count" -ne "${#seeds[@]}" ]]; then
  echo "Expected ${#seeds[@]} complete references, found $reference_count" >&2
  exit 1
fi

# Run one Habitat OpenGL context per declared GPU in each wave.
count=0
for seed in "${seeds[@]}"; do
  for spec in "unknown_gate unknown15" "novelty_gate novelty08"; do
    read -r policy label <<<"$spec"
    run_gate "$policy" "$label" "$seed" "${gpu_ids[$((count % ${#gpu_ids[@]}))]}" &
    count=$((count + 1))
    if (( count % ${#gpu_ids[@]} == 0 )); then
      wait
    fi
  done
done
wait
summary_count=$(find "$output_root" -maxdepth 2 -name summary.json | wc -l)
expected_count=$(( ${#seeds[@]} * 3 ))
if [[ "$summary_count" -ne "$expected_count" ]]; then
  echo "Expected $expected_count complete rollouts, found $summary_count" >&2
  exit 1
fi
"$python_bin" "$aggregator" "$output_root" "$output_root/aggregate"
