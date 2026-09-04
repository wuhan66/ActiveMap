#!/usr/bin/env bash
# Development-only matched-path sweep for the stateful RGB-D novelty gate.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 OUTPUT_ROOT SCENE_GLB SEED" >&2
  exit 2
fi

output_root=$1
scene=$2
seed=$3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${ACTIVEMAP_AGENT_PYTHON:-/home/wh/ActiveMap/envs/activemap-agent/bin/python}
runner="$repo_root/scripts/run_habitat_online_acquisition_pilot.py"
aggregator="$repo_root/scripts/aggregate_habitat_online_acquisition.py"

if [[ ! -f "$scene" ]]; then
  echo "Scene does not exist: $scene" >&2
  exit 2
fi
mkdir -p "$output_root"

reference="$output_root/acquire_all/committed_occupancy.npy"
if [[ ! -f "$reference" ]]; then
  "$python_bin" "$runner" "$output_root/acquire_all" \
    --scene "$scene" --policy acquire_all --gpu-id 0 --seed "$seed" --save-rgb \
    --goal-min-distance-m 1 --goal-max-distance-m 8
fi

launch() {
  local label=$1
  local policy=$2
  local threshold=$3
  local gpu=$4
  local output="$output_root/$label"
  [[ -d "$output" ]] && { echo "Refusing to overwrite $output" >&2; return 1; }
  if [[ "$policy" == "unknown_gate" ]]; then
    "$python_bin" "$runner" "$output" --scene "$scene" --policy "$policy" \
      --min-unknown-score "$threshold" --gpu-id "$gpu" --seed "$seed" \
      --reference-occupancy "$reference" --save-rgb \
      --goal-min-distance-m 1 --goal-max-distance-m 8 >"$output_root/$label.log" 2>&1
  else
    "$python_bin" "$runner" "$output" --scene "$scene" --policy "$policy" \
      --min-novelty-score "$threshold" --gpu-id "$gpu" --seed "$seed" \
      --reference-occupancy "$reference" --save-rgb \
      --goal-min-distance-m 1 --goal-max-distance-m 8 >"$output_root/$label.log" 2>&1
  fi
}

launch unknown15 unknown_gate 15 1 &
launch novelty06 novelty_gate 6 2 &
launch novelty08 novelty_gate 8 3 &
launch novelty10 novelty_gate 10 4 &
wait
"$python_bin" "$aggregator" "$output_root" "$output_root/aggregate"
