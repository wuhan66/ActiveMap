#!/usr/bin/env bash
set -euo pipefail

# Train-only throughput and parity probe for the V5 selector-state builder.
# This never reads test episodes and refuses to overwrite a prior probe.
if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 CHECKPOINT EPISODES OUTPUT_DIR" >&2
  exit 2
fi

checkpoint="$1"
episodes="$2"
output_dir="$3"
project_root="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${ACTIVEMAP_PYTHON:-python}"
gpu="${CUDA_VISIBLE_DEVICES:-0}"

[[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 1; }
[[ -f "$episodes" ]] || { echo "missing episodes: $episodes" >&2; exit 1; }
[[ ! -e "$output_dir" ]] || {
  echo "refusing to overwrite existing output: $output_dir" >&2
  exit 1
}

mkdir -p "$output_dir"
export PYTHONPATH="$project_root:$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"

cat >"$output_dir/probe_contract.json" <<EOF
{
  "schema_version": "sn7-v5-builder-throughput-probe-v1",
  "split": "train",
  "test_assets_read": false,
  "max_episodes": 5,
  "sequential_candidate_workers": 1,
  "parallel_candidate_workers": 8,
  "checkpoint": "$checkpoint",
  "episodes": "$episodes"
}
EOF

common_args=(
  -m activemap.cli build-selector-oracle "$checkpoint" "$episodes"
  --device cuda --image-size 128 --utility-mode executable --utility-profile balanced
  --cost-weight 0.18 --false-edit-weight 0.35 --budgets 1.5,3.0,4.5
  --initial-evidence-strategy min_cost --splits train --max-episodes 5
)

{
  /usr/bin/time -f "sequential_elapsed=%E" "$python_bin" "${common_args[@]}" \
    "$output_dir/sequential.jsonl" --candidate-workers 1
} >"$output_dir/sequential.log" 2>&1

{
  /usr/bin/time -f "parallel_elapsed=%E" "$python_bin" "${common_args[@]}" \
    "$output_dir/parallel.jsonl" --candidate-workers 8
} >"$output_dir/parallel.log" 2>&1

"$python_bin" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

from activemap.training.data import load_selector_samples

output_dir = Path(sys.argv[1])
sequential = [sample.model_dump() for sample in load_selector_samples(output_dir / "sequential.jsonl")]
parallel = [sample.model_dump() for sample in load_selector_samples(output_dir / "parallel.jsonl")]
if sequential != parallel:
    raise RuntimeError("parallel builder output differs from sequential output")
(output_dir / "PARITY_OK.json").write_text(
    json.dumps({"samples": len(sequential), "candidate_workers": 8}, indent=2) + "\n",
    encoding="utf-8",
)
PY

touch "$output_dir/COMPLETE"
echo "V5 builder throughput probe complete: $output_dir"
