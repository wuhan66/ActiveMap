#!/usr/bin/env bash
set -euo pipefail

# Registered, validation-only V5 factorial execution. This script is purposely
# fail-closed: it never reads test assets and creates a new root only after the
# three independent V5-B candidate-headroom receipts are hash-authorized.
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
DATA_ROOT="${SN7_V5_OUTPUT:-${STORAGE_ROOT}/processed/sn7_v1/updater_v5_temporal_pair_trainval_r1}"
EPISODES="${DATA_ROOT}/episodes_trainval_v5.jsonl"
BASE_SELECTOR_CONFIG="${PROJECT_ROOT}/configs/selector/sn7_v5_matched_v1_server.yaml"
RUN_ROOT="${V5_MATCHED_RUN_ROOT:-${STORAGE_ROOT}/runs/paper_evidence/sn7_v5_matched_nonkeep_2x2_v2}"
LOG_DIR="${STORAGE_ROOT}/logs"
SEEDS=(20260817 20260818 20260819)
GPUS=(0 1 2 3)
MODE="fresh"
case "${1:-}" in
  --resume)
    MODE="resume"
    shift
    ;;
  --resume-writeback)
    MODE="resume_writeback"
    shift
    ;;
  "")
    ;;
  *)
    echo "Usage: $0 [--resume|--resume-writeback]" >&2
    exit 2
    ;;
esac
[[ "$#" -eq 0 ]] || { echo "Usage: $0 [--resume|--resume-writeback]" >&2; exit 2; }

[[ -x "${PYTHON}" ]] || { echo "ActiveMap Python is missing: ${PYTHON}" >&2; exit 1; }
[[ -f "${EPISODES}" ]] || { echo "V5 train/validation episode manifest is missing" >&2; exit 1; }
[[ -f "${BASE_SELECTOR_CONFIG}" ]] || { echo "V5 selector config is missing" >&2; exit 1; }
if [[ "${MODE}" == "fresh" ]]; then
  [[ ! -e "${RUN_ROOT}" ]] || { echo "Refusing to overwrite V5 matched root: ${RUN_ROOT}" >&2; exit 1; }
else
  [[ -d "${RUN_ROOT}" ]] || { echo "V5 resume root is missing: ${RUN_ROOT}" >&2; exit 1; }
fi
for gpu in "${GPUS[@]}"; do
  active="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${active//[[:space:]]/}" ]] || { echo "GPU ${gpu} is occupied: ${active}" >&2; exit 1; }
done

mkdir -p "${RUN_ROOT}" "${LOG_DIR}"
exec 9>"${RUN_ROOT}/.queue.lock"
flock -n 9 || { echo "V5 matched queue is already active" >&2; exit 1; }
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

status_path="${RUN_ROOT}/queue_status.json"
write_status() {
  "${PYTHON}" - "$status_path" "$1" <<'PY'
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
if [[ "${MODE}" == "fresh" ]]; then
  write_status "starting"
elif [[ "${MODE}" == "resume" ]]; then
  "${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_sn7_v5_matched_resume.py" \
    "${RUN_ROOT}" --output "${RUN_ROOT}/resume_preflight.json" \
    >"${RUN_ROOT}/resume_preflight.log" 2>&1
  write_status "resuming"
else
  "${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_sn7_v5_matched_writeback_resume.py" \
    "${RUN_ROOT}" --output "${RUN_ROOT}/writeback_resume_preflight.json" \
    >"${RUN_ROOT}/writeback_resume_preflight.log" 2>&1
  write_status "resuming_writeback"
fi

AUTH_DIR="${RUN_ROOT}/authorization"
if [[ "${MODE}" == "fresh" ]]; then
  mkdir -p "${AUTH_DIR}/checkpoint_receipts"
  for seed in "${SEEDS[@]}"; do
    checkpoint="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${seed}/best_quality.pt"
    [[ -f "${checkpoint}" ]] || { echo "Missing V5-B checkpoint for seed ${seed}" >&2; exit 1; }
    "${PYTHON}" - "${checkpoint}" "${AUTH_DIR}/checkpoint_receipts/seed${seed}.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

checkpoint = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
output.write_text(
    json.dumps(
        {
            "schema_version": "sn7-v5b-checkpoint-receipt-v1",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "selection": "best_quality: predeclared temporal_change validation score",
            "test_assets_read": False,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
  done

  AUTHORIZATION="${AUTH_DIR}/three_seed_headroom_authorization.json"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/aggregate_sn7_v5b_headroom_replications.py" "${AUTHORIZATION}" \
    --record "20260817=${DATA_ROOT}/nonkeep_candidate_headroom_val_v5b/summary.json,${AUTH_DIR}/checkpoint_receipts/seed20260817.json" \
    --record "20260818=${DATA_ROOT}/nonkeep_candidate_headroom_val_v5b_seed20260818/summary.json,${AUTH_DIR}/checkpoint_receipts/seed20260818.json" \
    --record "20260819=${DATA_ROOT}/nonkeep_candidate_headroom_val_v5b_seed20260819/summary.json,${AUTH_DIR}/checkpoint_receipts/seed20260819.json" \
    >"${RUN_ROOT}/authorization.log" 2>&1
fi
AUTHORIZATION="${AUTH_DIR}/three_seed_headroom_authorization.json"
[[ -f "${AUTHORIZATION}" ]] || { echo "Missing V5 three-seed authorization" >&2; exit 1; }

state_root="${RUN_ROOT}/selector_states"
config_root="${RUN_ROOT}/selector_configs"
selector_root="${RUN_ROOT}/selectors"
mkdir -p "${state_root}" "${config_root}" "${selector_root}"

build_train_states() {
  local seed="$1" gpu="$2"
  local checkpoint="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${seed}/best_quality.pt"
  local output="${state_root}/seed${seed}_train.jsonl"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m activemap.cli build-selector-oracle \
    "${checkpoint}" "${EPISODES}" "${output}" \
    --device cuda --image-size 128 --utility-mode executable --utility-profile balanced \
    --cost-weight 0.18 --false-edit-weight 0.35 --budgets 1.5,3.0,4.5 \
    --initial-evidence-strategy min_cost --splits train \
    >"${LOG_DIR}/sn7_v5_matched_train_states_seed${seed}.log" 2>&1
}

if [[ "${MODE}" == "fresh" ]]; then
  pids=()
  for index in "${!SEEDS[@]}"; do
    build_train_states "${SEEDS[$index]}" "${GPUS[$index]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
fi

if [[ "${MODE}" != "resume_writeback" ]]; then
  for seed in "${SEEDS[@]}"; do
    train_states="${state_root}/seed${seed}_train.jsonl"
    [[ -s "${train_states}" ]] || { echo "Missing completed V5 train states for seed ${seed}" >&2; exit 1; }
    internal_dir="${state_root}/seed${seed}_internal"
    if [[ ! -d "${internal_dir}" ]]; then
      "${PYTHON}" "${PROJECT_ROOT}/scripts/split_sn7_v5_selector_train_internal.py" \
        "${train_states}" "${internal_dir}" --tune-fraction 0.20 --seed "${seed}" \
        >"${state_root}/seed${seed}_internal_split.log" 2>&1
    fi
    combined="${internal_dir}/fit_tune.jsonl"
    if [[ ! -f "${state_root}/seed${seed}_audit.json" ]]; then
      "${PYTHON}" "${PROJECT_ROOT}/scripts/audit_selector_states.py" "${combined}" \
        --output "${state_root}/seed${seed}_audit.json" >"${state_root}/seed${seed}_audit.log" 2>&1
    fi
    "${PYTHON}" "${PROJECT_ROOT}/scripts/render_sn7_v5_selector_config.py" \
      "${BASE_SELECTOR_CONFIG}" "${combined}" "${config_root}/seed${seed}.yaml" \
      "${selector_root}/seed${seed}" --seed "${seed}"
  done
fi

train_selector() {
  local seed="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -m activemap.cli train-selector \
    "${config_root}/seed${seed}.yaml" >"${LOG_DIR}/sn7_v5_matched_selector_seed${seed}.log" 2>&1
  [[ -f "${selector_root}/seed${seed}/best.pt" ]] || {
    echo "selector seed ${seed} did not produce best.pt" >&2
    return 1
  }
}

if [[ "${MODE}" != "resume_writeback" ]]; then
  pids=()
  for index in "${!SEEDS[@]}"; do
    train_selector "${SEEDS[$index]}" "${GPUS[$index]}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
fi

rollout_root="${RUN_ROOT}/rollouts"
mkdir -p "${rollout_root}"
if [[ "${MODE}" != "resume_writeback" ]]; then
for seed in "${SEEDS[@]}"; do
  checkpoint="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${seed}/best_quality.pt"
  selector="${selector_root}/seed${seed}/best.pt"
  train_states="${state_root}/seed${seed}_train.jsonl"
  val_states="${DATA_ROOT}/selector_states_headroom_val_v5b${seed:+_seed${seed}}.jsonl"
  if [[ "${seed}" == "20260817" ]]; then val_states="${DATA_ROOT}/selector_states_headroom_val_v5b.jsonl"; fi
  "${PYTHON}" "${PROJECT_ROOT}/scripts/build_sn7_v5_matched_rollouts.py" \
    "${train_states}" "${rollout_root}/seed${seed}_train" --authorization "${AUTHORIZATION}" \
    --updater-checkpoint "${checkpoint}" --updater-seed "${seed}" --selector-checkpoint "${selector}" \
    --split train --device cuda:0 >"${LOG_DIR}/sn7_v5_matched_rollouts_train_seed${seed}.log" 2>&1
  "${PYTHON}" "${PROJECT_ROOT}/scripts/build_sn7_v5_matched_rollouts.py" \
    "${val_states}" "${rollout_root}/seed${seed}_val" --authorization "${AUTHORIZATION}" \
    --updater-checkpoint "${checkpoint}" --updater-seed "${seed}" --selector-checkpoint "${selector}" \
    --split val --device cuda:0 >"${LOG_DIR}/sn7_v5_matched_rollouts_val_seed${seed}.log" 2>&1
done
fi

writeback_root="${RUN_ROOT}/writebacks"
mkdir -p "${writeback_root}"
run_raw_pair() {
  local seed="$1" split="$2" direct_gpu="$3" selected_gpu="$4"
  local checkpoint="${STORAGE_ROOT}/runs/updater/v5_temporal_explicit_change_trainval_r1_seed${seed}/best_quality.pt"
  local rollouts="${rollout_root}/seed${seed}_${split}"
  local seed_root="${writeback_root}/seed${seed}/${split}"
  mkdir -p "${seed_root}"
  CUDA_VISIBLE_DEVICES="${direct_gpu}" "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_agent_map_writeback.py" \
    "${checkpoint}" "${EPISODES}" "${rollouts}/direct_rollouts.jsonl" "${seed_root}/direct_commit" \
    --device cuda --split "${split}" --image-size 128 --threshold 0.5 \
    --protocol-name sn7-v5-temporal-matched-2x2-v1 \
    >"${LOG_DIR}/sn7_v5_direct_commit_${split}_seed${seed}.log" 2>&1 &
  local direct_pid=$!
  CUDA_VISIBLE_DEVICES="${selected_gpu}" "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_agent_map_writeback.py" \
    "${checkpoint}" "${EPISODES}" "${rollouts}/selected_rollouts.jsonl" "${seed_root}/selected_commit" \
    --device cuda --split "${split}" --image-size 128 --threshold 0.5 \
    --protocol-name sn7-v5-temporal-matched-2x2-v1 \
    >"${LOG_DIR}/sn7_v5_selected_commit_${split}_seed${seed}.log" 2>&1 &
  local selected_pid=$!
  wait "${direct_pid}"
  wait "${selected_pid}"
}

# Train writebacks are completed and calibrated before any validation writeback
# is created, preventing validation quality from influencing Safe Commit.
run_raw_pair "20260817" train "${GPUS[0]}" "${GPUS[1]}" & first_pair=$!
run_raw_pair "20260818" train "${GPUS[2]}" "${GPUS[3]}" & second_pair=$!
wait "${first_pair}"; wait "${second_pair}"
run_raw_pair "20260819" train "${GPUS[0]}" "${GPUS[1]}"

calibration_root="${RUN_ROOT}/train_calibration"
mkdir -p "${calibration_root}"
for seed in "${SEEDS[@]}"; do
  seed_root="${writeback_root}/seed${seed}/train"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/calibrate_sn7_v5_safe_commit.py" \
    "${calibration_root}/seed${seed}.json" \
    --record "direct=${seed_root}/direct_commit/writeback.jsonl" \
    --record "selected=${seed_root}/selected_commit/writeback.jsonl" \
    >"${LOG_DIR}/sn7_v5_safe_commit_calibration_seed${seed}.log" 2>&1
done

run_raw_pair "20260817" val "${GPUS[0]}" "${GPUS[1]}" & first_pair=$!
run_raw_pair "20260818" val "${GPUS[2]}" "${GPUS[3]}" & second_pair=$!
wait "${first_pair}"; wait "${second_pair}"
run_raw_pair "20260819" val "${GPUS[0]}" "${GPUS[1]}"

for seed in "${SEEDS[@]}"; do
  seed_root="${writeback_root}/seed${seed}/val"
  calibration="${calibration_root}/seed${seed}.json"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/apply_sn7_v5_safe_commit.py" \
    "${seed_root}/direct_commit/writeback.jsonl" "${calibration}" "${seed_root}/direct_safe_commit" \
    --policy direct >"${LOG_DIR}/sn7_v5_direct_safe_commit_seed${seed}.log" 2>&1
  "${PYTHON}" "${PROJECT_ROOT}/scripts/apply_sn7_v5_safe_commit.py" \
    "${seed_root}/selected_commit/writeback.jsonl" "${calibration}" "${seed_root}/selected_safe_commit" \
    --policy selected >"${LOG_DIR}/sn7_v5_selected_safe_commit_seed${seed}.log" 2>&1
done

aggregate_args=()
for seed in "${SEEDS[@]}"; do
  seed_root="${writeback_root}/seed${seed}/val"
  aggregate_args+=(--record "${seed}:direct_commit=${seed_root}/direct_commit/writeback.jsonl")
  aggregate_args+=(--record "${seed}:direct_safe_commit=${seed_root}/direct_safe_commit/writeback.jsonl")
  aggregate_args+=(--record "${seed}:selected_commit=${seed_root}/selected_commit/writeback.jsonl")
  aggregate_args+=(--record "${seed}:selected_safe_commit=${seed_root}/selected_safe_commit/writeback.jsonl")
done
"${PYTHON}" "${PROJECT_ROOT}/scripts/aggregate_sn7_v5_factorial_writebacks.py" \
  "${RUN_ROOT}/three_seed_nonkeep_factorial_summary.json" "${aggregate_args[@]}" \
  >"${RUN_ROOT}/aggregate.log" 2>&1
write_status "complete"
echo "V5 matched non-KEEP 2x2 complete: ${RUN_ROOT}"
