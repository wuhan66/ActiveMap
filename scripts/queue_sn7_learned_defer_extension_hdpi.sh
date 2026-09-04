#!/usr/bin/env bash
# Validation-only learned-defer extension; never opens the sealed SN7 test.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
BASE_ROOT="${BASE_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/iclr2027_r1_r2_validation_v1}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/iclr2027_learned_defer_extension_v1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/iclr2027_learned_defer_extension_v1}"
GPUS=(${GPUS:-2 3})
SEEDS=(20260730 20260731 20260801)

VAL_STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/states_val_step0.jsonl"
VAL_EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
REGISTRY="${PROJECT_ROOT}/configs/experiments/sn7_step0_frozen_registry_v3_cap20.yaml"

(( ${#GPUS[@]} >= 1 )) || { echo "at least one GPU is required" >&2; exit 2; }
for path in "$VAL_STATES" "$VAL_EPISODES" "$REGISTRY" "$BASE_ROOT/COMPLETE.json"; do
  test -f "$path" || { echo "missing validation input: $path" >&2; exit 3; }
done
[[ ! -e "$RUN_ROOT" ]] || { echo "refusing to reuse output root: $RUN_ROOT" >&2; exit 4; }

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT:${PYTHONPATH:-}"
mkdir -p "$RUN_ROOT" "$LOG_ROOT"
printf '%s\n' "validation-only learned-defer extension; test assets forbidden" > "$RUN_ROOT/PROTOCOL.txt"
sha256sum "$VAL_STATES" "$VAL_EPISODES" "$REGISTRY" "$BASE_ROOT/COMPLETE.json" > "$RUN_ROOT/input_sha256.txt"

run_seed() {
  local seed="$1"
  local gpu="$2"
  local selector="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${seed}/generic_utility_seed${seed}/best.pt"
  test -f "$selector" || { echo "missing generic selector: $selector" >&2; return 5; }
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/run_sn7_learned_defer_extension.py \
    "$VAL_STATES" "$VAL_EPISODES" "$REGISTRY" "$selector" "$seed" \
    "$BASE_ROOT" "$RUN_ROOT/seed${seed}" \
    --storage-root "$STORAGE_ROOT" --project-root "$PROJECT_ROOT" \
    --device cuda:0 --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --per-seed-bootstrap-repetitions 1 \
    > "$LOG_ROOT/seed${seed}.log" 2>&1
}

pids=()
for index in "${!SEEDS[@]}"; do
  gpu="${GPUS[$(( index % ${#GPUS[@]} ))]}"
  run_seed "${SEEDS[$index]}" "$gpu" &
  pids+=("$!")
  if (( ${#pids[@]} == ${#GPUS[@]} )); then
    for pid in "${pids[@]}"; do wait "$pid"; done
    pids=()
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PYTHON" scripts/aggregate_sn7_learned_defer_extension.py \
  "$RUN_ROOT" "$RUN_ROOT/three_seed_summary.json" \
  --bootstrap-repetitions 10000 --bootstrap-seed 20260830 \
  > "$LOG_ROOT/aggregate.log" 2>&1

"$PYTHON" - "$RUN_ROOT" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
summary = root / "three_seed_summary.json"
receipts = []
for seed in (20260730, 20260731, 20260801):
    path = root / f"seed{seed}" / "COMPLETE.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError(f"invalid learned-defer receipt for seed {seed}")
    receipts.append({"seed": seed, "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
(root / "COMPLETE.json").write_text(json.dumps({
    "schema_version": "activemap-sn7-learned-defer-extension-queue-v1",
    "split": "val",
    "test_assets_read": False,
    "seeds": receipts,
    "aggregate": {"path": str(summary), "sha256": hashlib.sha256(summary.read_bytes()).hexdigest()},
}, indent=2) + "\n", encoding="utf-8")
PY
echo "completed validation-only learned-defer extension"
