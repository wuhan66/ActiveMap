#!/usr/bin/env bash
# Validation-only reviewer-defense queue. It never opens the sealed SN7 test.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/iclr2027_r1_r2_validation_v1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/iclr2027_r1_r2_validation_v1}"
GPUS=(${GPUS:-1 3 4})
SEEDS=(20260730 20260731 20260801)

TRAIN_STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/states_train_step0.jsonl"
TRAIN_EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_train_bundle_v1/episodes_train.jsonl"
VAL_STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/states_val_step0.jsonl"
VAL_EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
REGISTRY="${PROJECT_ROOT}/configs/experiments/sn7_step0_frozen_registry_v3_cap20.yaml"

(( ${#GPUS[@]} >= ${#SEEDS[@]} )) || { echo "one GPU per frozen seed is required" >&2; exit 2; }
for path in "$TRAIN_STATES" "$TRAIN_EPISODES" "$VAL_STATES" "$VAL_EPISODES" "$REGISTRY"; do
  test -f "$path" || { echo "missing required validation input: $path" >&2; exit 3; }
done
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing to reuse output root: $RUN_ROOT" >&2; exit 4; }

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
printf '%s\n' "validation-only; test assets forbidden" > "$RUN_ROOT/PROTOCOL.txt"
sha256sum "$TRAIN_STATES" "$TRAIN_EPISODES" "$VAL_STATES" "$VAL_EPISODES" "$REGISTRY" > "$RUN_ROOT/input_sha256.txt"

run_seed() {
  local seed="$1"
  local gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_sn7_r1_r2_validation.py \
    "$TRAIN_STATES" "$TRAIN_EPISODES" "$VAL_STATES" "$VAL_EPISODES" \
    "$REGISTRY" "$seed" "$RUN_ROOT/seed${seed}" \
    --storage-root "$STORAGE_ROOT" --project-root "$PROJECT_ROOT" \
    --device cuda:0 --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --per-seed-bootstrap-repetitions 1 \
    > "$LOG_ROOT/seed${seed}.log" 2>&1
}

pids=()
for index in "${!SEEDS[@]}"; do
  run_seed "${SEEDS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
(( status == 0 )) || exit "$status"

"$PYTHON" scripts/aggregate_sn7_r1_r2_validation.py \
  "$RUN_ROOT" "$RUN_ROOT/three_seed_summary.json" \
  --bootstrap-repetitions 10000 --bootstrap-seed 20260813 \
  > "$LOG_ROOT/aggregate.log" 2>&1

"$PYTHON" - "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = (20260730, 20260731, 20260801)
receipts = []
for seed in seeds:
    path = root / f"seed{seed}" / "COMPLETE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError(f"invalid validation receipt for seed {seed}")
    receipts.append({"seed": seed, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(root / "COMPLETE.json").write_text(json.dumps({
    "schema_version": "activemap-iclr2027-r1-r2-validation-queue-v1",
    "split": "val",
    "test_assets_read": False,
    "seeds": receipts,
}, indent=2) + "\n", encoding="utf-8")
PY
echo "completed validation-only R1/R2 queue"
