#!/usr/bin/env bash
set -euo pipefail

# V6 executable continuation. This queue is intentionally separate from the
# completed V5 factorial root: it keeps V5's updater, candidate interface, and
# typed writeback evaluator frozen while replacing only the candidate scorer.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
EPISODES="${DATA_ROOT}/episodes_trainval_v5.jsonl"
V5_ROOT="${V5_MATCHED_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_matched_nonkeep_2x2_v2}"
V6_VALUE_ROOT="${V6_EVIDENCE_VALUE_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v6_evidence_value_recovery_v1}"
V6_AUDIT_DIR="${V6_EVIDENCE_VALUE_AUDIT_DIR:-policy_audit_val_final_v2}"
RUN_ROOT="${V6_EVIDENCE_VALUE_WRITEBACK_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v6_evidence_value_writeback_v3_final}"
LOG_DIR="${STORAGE_ROOT}/logs/sn7_v6_evidence_value_writeback_v3_final"
SEEDS=(20260817 20260818 20260819)
GPUS=(0 1 2)
MODE="run"
case "${1:-}" in
  --preflight)
    MODE="preflight"
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [--preflight]" >&2
    exit 2
    ;;
esac

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ -f "${EPISODES}" ]] || { echo "V5 train/validation episodes are missing" >&2; exit 1; }
[[ -d "${V5_ROOT}" ]] || { echo "V5 matched root is missing" >&2; exit 1; }
[[ -d "${V6_VALUE_ROOT}" ]] || { echo "V6 Evidence Value root is missing" >&2; exit 1; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite V6 writeback root: ${RUN_ROOT}" >&2; exit 1; }
for gpu in "${GPUS[@]}"; do
  active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${gpu} is occupied: ${active}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
exec 9>"${RUN_ROOT}/.queue.lock"
flock -n 9 || { echo "V6 writeback queue is already active" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON}" - "${V6_VALUE_ROOT}" "${V6_AUDIT_DIR}" "${RUN_ROOT}/protocol.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

value_root = Path(sys.argv[1])
audit_dir = sys.argv[2]
records = []
for seed in (20260817, 20260818, 20260819):
    summary_path = value_root / f"seed{seed}" / audit_dir / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "sn7-v6-evidence-value-policy-audit-v1":
        raise ValueError(f"{summary_path}: unexpected V6 policy-audit schema")
    if payload.get("split") != "val" or payload.get("test_assets_read") is not False:
        raise ValueError(f"{summary_path}: audit is not validation-only")
    checkpoint = Path(str(payload.get("checkpoint", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{summary_path}: checkpoint is missing")
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if checkpoint_sha256 != payload.get("checkpoint_sha256"):
        raise ValueError(
            f"{summary_path}: audited checkpoint hash does not match current file"
        )
    overall = payload.get("overall", {})
    if not 0.0 < float(overall.get("acquire_rate", 0.0)) < 1.0:
        raise ValueError(f"{summary_path}: V6 acquisition collapsed or saturated")
    if float(overall.get("false_call_rate", 1.0)) > 0.05:
        raise ValueError(f"{summary_path}: V6 exceeds its train-only false-call constraint")
    records.append(
        {
            "seed": seed,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "policy_audit": str(summary_path.resolve()),
            "policy_audit_summary": overall,
        }
    )

Path(sys.argv[3]).write_text(
    json.dumps(
        {
            "schema_version": "sn7-v6-evidence-value-writeback-v1",
            "seeds": records,
            "train_only_safe_commit_calibration": True,
            "formal_validation_used_for_policy_tuning": False,
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY

if [[ "${MODE}" == "preflight" ]]; then
  echo "V6 evidence-value writeback preflight passed: ${RUN_ROOT}"
  exit 0
fi

AUTHORIZATION="${V5_ROOT}/authorization/three_seed_headroom_authorization.json"
[[ -f "${AUTHORIZATION}" ]] || { echo "Missing V5 headroom authorization" >&2; exit 1; }

state_path() {
  local seed="$1" split="$2"
  if [[ "${split}" == "train" ]]; then
    printf '%s\n' "${V5_ROOT}/selector_states/seed${seed}_train.jsonl"
  elif [[ "${seed}" == "20260817" ]]; then
    printf '%s\n' "${DATA_ROOT}/selector_states_headroom_val_v5b.jsonl"
  else
    printf '%s\n' "${DATA_ROOT}/selector_states_headroom_val_v5b_seed${seed}.jsonl"
  fi
}

build_rollouts() {
  local seed="$1" split="$2" gpu="$3"
  local states
  states="$(state_path "${seed}" "${split}")"
  local updater="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${seed}/best_quality.pt"
  local value_head="${V6_VALUE_ROOT}/seed${seed}/run/best.pt"
  local output="${RUN_ROOT}/rollouts/seed${seed}_${split}"
  [[ -f "${states}" ]] || { echo "Missing ${split} states: ${states}" >&2; exit 1; }
  [[ -f "${updater}" && -f "${value_head}" ]] || { echo "Missing frozen scorer or updater for ${seed}" >&2; exit 1; }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${PROJECT_ROOT}/scripts/build_sn7_v5_matched_rollouts.py" \
    "${states}" "${output}" --authorization "${AUTHORIZATION}" \
    --updater-checkpoint "${updater}" --updater-seed "${seed}" \
    --evidence-value-checkpoint "${value_head}" --split "${split}" --device cuda \
    >"${LOG_DIR}/build_${split}_seed${seed}.log" 2>&1
}

evaluate_policy() {
  local seed="$1" split="$2" policy="$3" gpu="$4"
  local updater="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${seed}/best_quality.pt"
  local rollout="${RUN_ROOT}/rollouts/seed${seed}_${split}/${policy}_rollouts.jsonl"
  local output="${RUN_ROOT}/raw_writebacks/seed${seed}/${split}/${policy}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_agent_map_writeback.py" \
    "${updater}" "${EPISODES}" "${rollout}" "${output}" --device cuda --split "${split}" \
    --image-size 128 --threshold 0.5 --protocol-name sn7-v6-evidence-value-matched-writeback-v1 \
    >"${LOG_DIR}/writeback_${split}_${policy}_seed${seed}.log" 2>&1
}

evaluate_triplet() {
  local seed="$1" split="$2"
  evaluate_policy "${seed}" "${split}" direct "${GPUS[0]}" & direct_pid=$!
  evaluate_policy "${seed}" "${split}" selected "${GPUS[1]}" & selected_pid=$!
  evaluate_policy "${seed}" "${split}" forced "${GPUS[2]}" & forced_pid=$!
  wait "${direct_pid}"
  wait "${selected_pid}"
  wait "${forced_pid}"
}

for index in "${!SEEDS[@]}"; do
  build_rollouts "${SEEDS[$index]}" train "${GPUS[$index]}" &
done
wait

for seed in "${SEEDS[@]}"; do
  evaluate_triplet "${seed}" train
done

mkdir -p "${RUN_ROOT}/train_calibration"
for seed in "${SEEDS[@]}"; do
  root="${RUN_ROOT}/raw_writebacks/seed${seed}/train"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/calibrate_sn7_v5_safe_commit.py" \
    "${RUN_ROOT}/train_calibration/seed${seed}.json" \
    --record "direct=${root}/direct/writeback.jsonl" \
    --record "selected=${root}/selected/writeback.jsonl" \
    >"${LOG_DIR}/calibrate_safe_commit_seed${seed}.log" 2>&1
done

for index in "${!SEEDS[@]}"; do
  build_rollouts "${SEEDS[$index]}" val "${GPUS[$index]}" &
done
wait

for seed in "${SEEDS[@]}"; do
  evaluate_triplet "${seed}" val
done

for seed in "${SEEDS[@]}"; do
  root="${RUN_ROOT}/raw_writebacks/seed${seed}/val"
  calibration="${RUN_ROOT}/train_calibration/seed${seed}.json"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/apply_sn7_v5_safe_commit.py" \
    "${root}/direct/writeback.jsonl" "${calibration}" "${RUN_ROOT}/safe_writebacks/seed${seed}/direct" \
    --policy direct >"${LOG_DIR}/safe_commit_direct_seed${seed}.log" 2>&1
  "${PYTHON}" "${PROJECT_ROOT}/scripts/apply_sn7_v5_safe_commit.py" \
    "${root}/selected/writeback.jsonl" "${calibration}" "${RUN_ROOT}/safe_writebacks/seed${seed}/selected" \
    --policy selected >"${LOG_DIR}/safe_commit_selected_seed${seed}.log" 2>&1
done

aggregate_args=()
for seed in "${SEEDS[@]}"; do
  raw="${RUN_ROOT}/raw_writebacks/seed${seed}/val"
  safe="${RUN_ROOT}/safe_writebacks/seed${seed}"
  aggregate_args+=(--record "${seed}:direct_commit=${raw}/direct/writeback.jsonl")
  aggregate_args+=(--record "${seed}:direct_safe_commit=${safe}/direct/writeback.jsonl")
  aggregate_args+=(--record "${seed}:selected_commit=${raw}/selected/writeback.jsonl")
  aggregate_args+=(--record "${seed}:selected_safe_commit=${safe}/selected/writeback.jsonl")
done
"${PYTHON}" "${PROJECT_ROOT}/scripts/aggregate_sn7_v5_factorial_writebacks.py" \
  "${RUN_ROOT}/three_seed_factorial_summary.json" "${aggregate_args[@]}" \
  >"${LOG_DIR}/aggregate.log" 2>&1

"${PYTHON}" - "${RUN_ROOT}/queue_status.json" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps(
        {
            "status": "complete",
            "next_stage": "review_executable_validation_before_any_test_access",
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
echo "V6 evidence-value executable writeback complete: ${RUN_ROOT}"
