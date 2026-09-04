#!/usr/bin/env bash
set -euo pipefail

ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
REPO="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ROOT}/envs/activemap-agent/bin/python"
RUN="${ROOT}/runs/sn7_active_catalog"
CAUSAL="${RUN}/final_tool_belief_causal_v1"
EPISODES="${ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full/closed_loop_v1/episodes_val.jsonl"
OUTPUT="${RUN}/final_chronological_maintenance_v1"
POLL_SECONDS="${POLL_SECONDS:-60}"
SEEDS=(20260717 20260718 20260719)
THRESHOLDS=(0.5 0.7 0.9)

cd "${REPO}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

until [[ -s "${CAUSAL}/manifest.json" ]]; do sleep "${POLL_SECONDS}"; done
[[ ! -e "${OUTPUT}" ]] || {
  echo "refusing existing chronological output: ${OUTPUT}" >&2
  exit 3
}

for threshold in "${THRESHOLDS[@]}"; do
  label="${threshold/./p}"
  for seed in "${SEEDS[@]}"; do
    source="${CAUSAL}/seed${seed}/selective_writeback/evaluation/writeback.jsonl"
    [[ -s "${source}" ]] || { echo "missing final writeback: ${source}" >&2; exit 4; }
    "${PYTHON}" scripts/evaluate_chronological_map_maintenance.py \
      "${EPISODES}" "${source}" "${OUTPUT}/threshold_${label}/seed${seed}" \
      --split val --budget 3.0 --minimum-chain-length 2 \
      --confidence-threshold "${threshold}" --replay-iou-threshold 0.99 \
      --bootstrap-repetitions 2000 --seed "${seed}"
  done
  records=()
  for seed in "${SEEDS[@]}"; do
    records+=(--record "${seed}=${OUTPUT}/threshold_${label}/seed${seed}/chronological_traces.jsonl")
  done
  "${PYTHON}" scripts/aggregate_chronological_map_maintenance.py \
    "${OUTPUT}/threshold_${label}/three_seed.json" \
    "${records[@]}" --bootstrap-repetitions 5000 --seed 20260729
done

"${PYTHON}" scripts/render_chronological_map_maintenance.py \
  "${OUTPUT}/threshold_0p7/seed20260718/chronological_traces.jsonl" \
  "${ROOT}/runs/paper_visuals/sn7_chronological_maintenance_v1" \
  --size 512 --maximum-chains 8

"${PYTHON}" - "${OUTPUT}/manifest.json" "${OUTPUT}" <<'PY'
import hashlib
import json
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])
sources = sorted(root.glob("threshold_*/three_seed.json"))
payload = {
    "schema_version": "sn7-final-chronological-maintenance-manifest-v1",
    "primary_confidence_threshold": 0.7,
    "sensitivity_thresholds": [0.5, 0.9],
    "budget": 3.0,
    "split": "val",
    "test_assets_read": False,
    "online_weight_update": False,
    "artifacts": {
        path.parent.name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sources
    },
}
output.write_text(json.dumps(payload, indent=2) + "\n")
PY
