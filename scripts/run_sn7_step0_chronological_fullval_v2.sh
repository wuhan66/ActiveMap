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
VARIANTS=(notool forced benefit)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
test -f "${EPISODES}"
[[ ! -e "${OUTPUT_ROOT}" ]] || {
  echo "refusing existing output: ${OUTPUT_ROOT}" >&2
  exit 41
}
mkdir -p "${OUTPUT_ROOT}/threshold_0p7" "${LOG_ROOT}"

for variant in "${VARIANTS[@]}"; do
  pids=()
  for seed in "${SEEDS[@]}"; do
    source="${WRITEBACK_ROOT}/seed${seed}/${variant}/writeback/evaluation/writeback.jsonl"
    output="${OUTPUT_ROOT}/threshold_0p7/${variant}/seed${seed}"
    test -s "${source}"
    "${PYTHON}" scripts/evaluate_chronological_map_maintenance.py \
      "${EPISODES}" "${source}" "${output}" \
      --split val --budget 3.0 --minimum-chain-length 2 \
      --continuity-tolerance 1e-6 \
      --confidence-threshold 0.7 --replay-iou-threshold 0.99 \
      --bootstrap-repetitions 0 --seed "${seed}" \
      >"${LOG_ROOT}/${variant}_seed${seed}.log" 2>&1 &
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
      "${seed}=${OUTPUT_ROOT}/threshold_0p7/${variant}/seed${seed}/chronological_traces.jsonl"
    )
  done
  "${PYTHON}" scripts/aggregate_chronological_map_maintenance.py \
    "${OUTPUT_ROOT}/threshold_0p7/${variant}/three_seed.json" \
    "${records[@]}" --bootstrap-repetitions 10000 --seed 20260729 \
    >"${LOG_ROOT}/${variant}_aggregate.log" 2>&1
done

"${PYTHON}" scripts/render_chronological_map_maintenance.py \
  "${OUTPUT_ROOT}/threshold_0p7/benefit/seed20260730/chronological_traces.jsonl" \
  "${STORAGE_ROOT}/runs/paper_visuals/sn7_step0_chronological_fullval_v2" \
  --size 512 --maximum-chains 8 \
  >"${LOG_ROOT}/render.log" 2>&1

"${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
artifacts = {}
for path in sorted(root.glob("threshold_0p7/*/three_seed.json")):
    artifacts[path.parent.name] = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
payload = {
    "schema_version": "sn7-step0-chronological-fullval-manifest-v2",
    "split": "val",
    "test_assets_read": False,
    "budget": 3.0,
    "confidence_threshold": 0.7,
    "continuity_tolerance": 1e-6,
    "minimum_chain_length": 2,
    "seeds": [20260730, 20260731, 20260801],
    "variants": ["notool", "forced", "benefit"],
    "artifacts": artifacts,
}
(root / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
PY

echo "completed SN7 Step-0 chronological full-validation v2"
