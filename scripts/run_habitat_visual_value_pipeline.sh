#!/usr/bin/env bash
# Development-only, scene-disjoint RGB-D visual-value pipeline for Habitat.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 OUTPUT_ROOT" >&2
  exit 2
fi

output_root=$1
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
storage_root=${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}
python_bin=${ACTIVEMAP_AGENT_PYTHON:-$storage_root/envs/activemap-agent/bin/python}
suite=${ACTIVEMAP_HABITAT_SUITE:-$storage_root/runs/robotics/habitat_scene_disjoint_suite_20260830_v2_24x24}
episodes="$output_root/episodes"
training="$output_root/training_seed20270831"
validation="$output_root/validation_budget2"

[[ ! -e "$output_root" ]] || { echo "Refusing to overwrite: $output_root" >&2; exit 2; }
[[ -f "$suite/suite_summary.json" ]] || { echo "Missing suite: $suite" >&2; exit 2; }
mkdir -p "$output_root"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

"$python_bin" "$repo_root/scripts/materialize_habitat_scene_disjoint_suite.py" \
  "$suite" "$episodes" --resolution-m 0.05 --max-range-m 5.0 --ray-stride 4 --budget 8.0 \
  --splits train val

"$python_bin" "$repo_root/scripts/train_habitat_visual_value.py" \
  "$episodes/manifests/train.jsonl" "$episodes/manifests/val.jsonl" "$training" \
  --device cuda:0 --seed 20270831 --epochs 80 --patience 12 --batch-size 32 \
  --cost-weight 0.0 --stop-margin 0.0 --target-mode absolute_gain

"$python_bin" "$repo_root/scripts/evaluate_habitat_visual_value.py" \
  "$episodes/manifests/val.jsonl" "$training/best.pt" "$validation" \
  --device cuda:0 --budget 2.0 --max-steps 2 --commit-rule always \
  --prior-override-margin 0.0 --prior-override-margin 0.001 \
  --prior-override-margin 0.002 --prior-override-margin 0.005 \
  --prior-override-margin 0.01 --prior-override-margin 0.02

echo "Wrote development-only visual-value pipeline: $output_root"
