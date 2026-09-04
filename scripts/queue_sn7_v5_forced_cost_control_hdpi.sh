#!/usr/bin/env bash
set -euo pipefail

# Registered before formal V5 validation writeback completes. This sidecar
# preserves the primary 2x2 matrix and adds only the matched cost control.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
EPISODES="${DATA_ROOT}/episodes_trainval_v5.jsonl"
RUN_ROOT="${V5_MATCHED_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_matched_nonkeep_2x2_v2}"
LOG_DIR="${STORAGE_ROOT}/logs"
SEEDS=(20260817 20260818 20260819)
GPUS=(0 1 2)
STATUS_PATH="${RUN_ROOT}/forced_cost_control_status.json"

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ -f "${EPISODES}" ]] || { echo "V5 train/validation episode manifest is missing" >&2; exit 1; }
[[ -d "${RUN_ROOT}" ]] || { echo "V5 matched run root is missing: ${RUN_ROOT}" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

write_status() {
  "${PYTHON}" - "${STATUS_PATH}" "$1" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"status": sys.argv[2], "split": "train,val", "test_assets_read": False}, indent=2)
    + "\n",
    encoding="utf-8",
)
PY
}

write_status "waiting_for_primary_matrix"
while true; do
  [[ -f "${RUN_ROOT}/queue_status.json" ]] || { sleep 120; continue; }
  primary_status="$("${PYTHON}" - "${RUN_ROOT}/queue_status.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("status", ""))
PY
)"
  [[ "${primary_status}" == "complete" ]] && break
  sleep 120
done

for seed in "${SEEDS[@]}"; do
  [[ -f "${RUN_ROOT}/rollouts/seed${seed}_val/forced_rollouts.jsonl" ]] || {
    echo "Forced rollout is absent for seed ${seed}; refusing post-hoc fallback" >&2
    exit 1
  }
  [[ -f "${RUN_ROOT}/train_calibration/seed${seed}.json" ]] || {
    echo "Train-only Safe Commit receipt is absent for seed ${seed}" >&2
    exit 1
  }
done

write_status "evaluating_forced_control"
run_forced() {
  local seed="$1" gpu="$2"
  local checkpoint="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${seed}/best_quality.pt"
  local seed_root="${RUN_ROOT}/writebacks/seed${seed}/val"
  local rollout="${RUN_ROOT}/rollouts/seed${seed}_val/forced_rollouts.jsonl"
  local calibration="${RUN_ROOT}/train_calibration/seed${seed}.json"
  [[ ! -e "${seed_root}/forced_commit" ]] || { echo "Forced raw output exists for ${seed}" >&2; return 1; }
  [[ ! -e "${seed_root}/forced_safe_commit" ]] || { echo "Forced gated output exists for ${seed}" >&2; return 1; }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_agent_map_writeback.py" \
    "${checkpoint}" "${EPISODES}" "${rollout}" "${seed_root}/forced_commit" \
    --device cuda --split val --image-size 128 --threshold 0.5 \
    --protocol-name sn7-v5-temporal-matched-forced-cost-control-v1 \
    >"${LOG_DIR}/sn7_v5_forced_commit_val_seed${seed}.log" 2>&1
  "${PYTHON}" "${PROJECT_ROOT}/scripts/apply_sn7_v5_safe_commit.py" \
    "${seed_root}/forced_commit/writeback.jsonl" "${calibration}" "${seed_root}/forced_safe_commit" \
    --policy forced >"${LOG_DIR}/sn7_v5_forced_safe_commit_seed${seed}.log" 2>&1
}

pids=()
for index in "${!SEEDS[@]}"; do
  run_forced "${SEEDS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

aggregate_args=()
for seed in "${SEEDS[@]}"; do
  seed_root="${RUN_ROOT}/writebacks/seed${seed}/val"
  aggregate_args+=(--record "${seed}:direct_commit=${seed_root}/direct_commit/writeback.jsonl")
  aggregate_args+=(--record "${seed}:direct_safe_commit=${seed_root}/direct_safe_commit/writeback.jsonl")
  aggregate_args+=(--record "${seed}:selected_commit=${seed_root}/selected_commit/writeback.jsonl")
  aggregate_args+=(--record "${seed}:selected_safe_commit=${seed_root}/selected_safe_commit/writeback.jsonl")
  aggregate_args+=(--record "${seed}:forced_safe_commit=${seed_root}/forced_safe_commit/writeback.jsonl")
done
output="${RUN_ROOT}/three_seed_nonkeep_factorial_with_forced_summary.json"
[[ ! -e "${output}" ]] || { echo "Forced-control aggregate exists" >&2; exit 1; }
"${PYTHON}" "${PROJECT_ROOT}/scripts/aggregate_sn7_v5_factorial_writebacks.py" \
  "${output}" "${aggregate_args[@]}" >"${RUN_ROOT}/aggregate_with_forced.log" 2>&1
write_status "complete"
echo "V5 forced-acquisition cost control complete: ${RUN_ROOT}"
