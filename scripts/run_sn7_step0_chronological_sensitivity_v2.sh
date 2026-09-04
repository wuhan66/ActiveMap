#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
WRITEBACK_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1"
OUTPUT_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog/step0_chronological_maintenance_fullval_v2"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_step0_chronological_fullval_v2"
SEEDS=(20260730 20260731 20260801)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
test -f "${OUTPUT_ROOT}/manifest.json"

for threshold in 0.5 0.9; do
  label="${threshold/./p}"
  output="${OUTPUT_ROOT}/threshold_${label}/benefit"
  [[ ! -e "${output}" ]] || {
    echo "refusing existing sensitivity output: ${output}" >&2
    exit 41
  }
  pids=()
  for seed in "${SEEDS[@]}"; do
    source="${WRITEBACK_ROOT}/seed${seed}/benefit/writeback/evaluation/writeback.jsonl"
    "${PYTHON}" scripts/evaluate_chronological_map_maintenance.py \
      "${EPISODES}" "${source}" "${output}/seed${seed}" \
      --split val --budget 3.0 --minimum-chain-length 2 \
      --continuity-tolerance 1e-6 \
      --confidence-threshold "${threshold}" --replay-iou-threshold 0.99 \
      --bootstrap-repetitions 0 --seed "${seed}" \
      >"${LOG_ROOT}/benefit_threshold_${label}_seed${seed}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  (( status == 0 )) || exit "${status}"
  records=()
  for seed in "${SEEDS[@]}"; do
    records+=(
      --record
      "${seed}=${output}/seed${seed}/chronological_traces.jsonl"
    )
  done
  "${PYTHON}" scripts/aggregate_chronological_map_maintenance.py \
    "${output}/three_seed_v2.json" "${records[@]}" \
    --bootstrap-repetitions 10000 --seed 20260729 \
    >"${LOG_ROOT}/benefit_threshold_${label}_aggregate.log" 2>&1
done

for variant in notool forced benefit; do
  records=()
  for seed in "${SEEDS[@]}"; do
    records+=(
      --record
      "${seed}=${OUTPUT_ROOT}/threshold_0p7/${variant}/seed${seed}/chronological_traces.jsonl"
    )
  done
  "${PYTHON}" scripts/aggregate_chronological_map_maintenance.py \
    "${OUTPUT_ROOT}/threshold_0p7/${variant}/three_seed_v2.json" \
    "${records[@]}" --bootstrap-repetitions 10000 --seed 20260729 \
    >"${LOG_ROOT}/${variant}_aggregate_v2.log" 2>&1
done

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
paths = sorted(root.glob("threshold_*/*/three_seed_v2.json"))
payload = {
    "schema_version": "sn7-step0-chronological-sensitivity-manifest-v2",
    "split": "val",
    "test_assets_read": False,
    "primary_threshold": 0.7,
    "sensitivity_thresholds": [0.5, 0.9],
    "support": {
        "chains_per_seed": 173,
        "transitions_per_seed": 411,
        "aoi_count": 9,
    },
    "artifacts": {
        str(path.relative_to(root)): {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }
        for path in paths
    },
}
(root / "sensitivity_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
PY

echo "completed SN7 Step-0 chronological sensitivity v2"
