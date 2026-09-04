#!/usr/bin/env bash
set -euo pipefail

# Validation-only true sequential controller extension. This script never
# accepts a test episode file and refuses to run without a hash-checked online
# controller registry.

STAGE="${1:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
VAL_EPISODES="${VAL_EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl}"
ONLINE_REGISTRY="${ONLINE_REGISTRY:-}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_true_sequential_four_policy_20260831}"
SAFE_CONFIDENCE_THRESHOLD="${SAFE_CONFIDENCE_THRESHOLD:-0.70}"
SAFE_REPLAY_IOU_THRESHOLD="${SAFE_REPLAY_IOU_THRESHOLD:-0.99}"
MASK_THRESHOLD="${MASK_THRESHOLD:-0.50}"
DELTA_MARGIN="${DELTA_MARGIN:-0.15}"
MIN_DELTA_COMPONENT_PIXELS="${MIN_DELTA_COMPONENT_PIXELS:-0}"
TOOL_GATE_THRESHOLD_OVERRIDE="${TOOL_GATE_THRESHOLD_OVERRIDE:-}"
CHAIN_INDEX="${CHAIN_INDEX:-0}"
CONTROLLER_SEEDS=(${CONTROLLER_SEEDS:-20260831 20260901 20260902})
GPU_IDS=(${GPU_IDS:-1 2 3})
ASSET_MAP="${ASSET_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"
POLICIES=(
  direct_current_hypothesis
  direct_current_hypothesis_safe
  active_selective_safe
  active_forced_safe
)

usage() {
  cat <<'EOF'
Usage:
  run_sn7_true_sequential_four_policy_hdpi.sh smoke
  run_sn7_true_sequential_four_policy_hdpi.sh pilot20
  run_sn7_true_sequential_four_policy_hdpi.sh full

Required environment:
  ONLINE_REGISTRY=/absolute/path/to/online-observable-registry.yaml

Optional environment:
  RUN_ROOT, CONTROLLER_SEEDS, GPU_IDS, SAFE_CONFIDENCE_THRESHOLD,
  SAFE_REPLAY_IOU_THRESHOLD, CHAIN_INDEX, PROJECT_ROOT, STORAGE_ROOT, PYTHON.
EOF
}

require_inputs() {
  [[ -n "${ONLINE_REGISTRY}" ]] || {
    echo "ONLINE_REGISTRY must name the frozen online-observable registry" >&2
    exit 2
  }
  [[ -f "${ONLINE_REGISTRY}" ]] || { echo "missing registry: ${ONLINE_REGISTRY}" >&2; exit 2; }
  [[ -f "${VAL_EPISODES}" ]] || { echo "missing validation episodes: ${VAL_EPISODES}" >&2; exit 2; }
  [[ -x "${PYTHON}" ]] || { echo "missing python: ${PYTHON}" >&2; exit 2; }
  if [[ "${STAGE}" == "full" ]]; then
    [[ ${#CONTROLLER_SEEDS[@]} -eq 3 ]] || { echo "exactly three controller seeds required" >&2; exit 2; }
    [[ ${#GPU_IDS[@]} -ge 3 ]] || { echo "at least three GPU ids required" >&2; exit 2; }
  else
    [[ ${#CONTROLLER_SEEDS[@]} -ge 1 ]] || { echo "at least one controller seed required" >&2; exit 2; }
    [[ ${#GPU_IDS[@]} -ge 1 ]] || { echo "at least one GPU id required" >&2; exit 2; }
  fi
}

require_new_smoke_root() {
  [[ ! -e "${RUN_ROOT}" ]] || {
    echo "refusing to overwrite smoke run root: ${RUN_ROOT}" >&2
    exit 2
  }
}

require_completed_smoke() {
  local smoke_output="${RUN_ROOT}/smoke_seed${CONTROLLER_SEEDS[0]}"
  [[ -d "${RUN_ROOT}" ]] || {
    echo "full stage requires a completed smoke root: ${RUN_ROOT}" >&2
    exit 2
  }
  [[ -f "${smoke_output}/online_full_controller_traces.jsonl" ]] || {
    echo "full stage requires smoke traces: ${smoke_output}" >&2
    exit 2
  }
  [[ -f "${smoke_output}/true_sequential_rollout_audit.json" ]] || {
    echo "full stage requires a smoke audit: ${smoke_output}" >&2
    exit 2
  }
  for seed in "${CONTROLLER_SEEDS[@]}"; do
    [[ ! -e "${RUN_ROOT}/seed${seed}" ]] || {
      echo "refusing to overwrite full seed output: ${RUN_ROOT}/seed${seed}" >&2
      exit 2
    }
  done
  [[ ! -e "${RUN_ROOT}/three_seed_horizon_summary.json" ]] || {
    echo "refusing to overwrite aggregate: ${RUN_ROOT}/three_seed_horizon_summary.json" >&2
    exit 2
  }
}

policy_args=()
for policy in "${POLICIES[@]}"; do
  policy_args+=(--policy "${policy}")
done

tool_gate_args=()
if [[ -n "${TOOL_GATE_THRESHOLD_OVERRIDE}" ]]; then
  tool_gate_args+=(
    --diagnostic-tool-gate-threshold-override "${TOOL_GATE_THRESHOLD_OVERRIDE}"
  )
fi

run_one() {
  local seed="$1"
  local gpu="$2"
  local output="$3"
  shift 3
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" scripts/evaluate_online_full_controller.py \
    "${VAL_EPISODES}" "${ONLINE_REGISTRY}" "${seed}" "${output}" \
    --storage-root "${STORAGE_ROOT}" --project-root "${PROJECT_ROOT}" --device cuda:0 \
    --image-size 512 --budget 3.0 --max-candidates 16 --max-acquisitions 2 --max-tool-calls 4 \
    --minimum-chain-length 2 --safe-confidence-threshold "${SAFE_CONFIDENCE_THRESHOLD}" \
    --safe-replay-iou-threshold "${SAFE_REPLAY_IOU_THRESHOLD}" \
    --threshold "${MASK_THRESHOLD}" --delta-margin "${DELTA_MARGIN}" \
    --min-delta-component-pixels "${MIN_DELTA_COMPONENT_PIXELS}" \
    --asset-root-map "${ASSET_MAP}" "${tool_gate_args[@]}" "${policy_args[@]}" "$@"
}

case "${STAGE}" in
  smoke)
    require_inputs
    require_new_smoke_root
    mkdir -p "${RUN_ROOT}/logs"
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
    run_one "${CONTROLLER_SEEDS[0]}" "${GPU_IDS[0]}" "${RUN_ROOT}/smoke_seed${CONTROLLER_SEEDS[0]}" \
      --chain-index "${CHAIN_INDEX}" \
      >"${RUN_ROOT}/logs/smoke_seed${CONTROLLER_SEEDS[0]}.log" 2>&1
    ;;
  pilot20)
    require_inputs
    require_new_smoke_root
    mkdir -p "${RUN_ROOT}/logs"
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
    run_one "${CONTROLLER_SEEDS[0]}" "${GPU_IDS[0]}" "${RUN_ROOT}/pilot20_seed${CONTROLLER_SEEDS[0]}" \
      --max-chains 20 \
      >"${RUN_ROOT}/logs/pilot20_seed${CONTROLLER_SEEDS[0]}.log" 2>&1
    ;;
  full)
    require_inputs
    require_completed_smoke
    mkdir -p "${RUN_ROOT}/logs"
    cd "${PROJECT_ROOT}"
    export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
    pids=()
    for index in "${!CONTROLLER_SEEDS[@]}"; do
      seed="${CONTROLLER_SEEDS[$index]}"
      gpu="${GPU_IDS[$index]}"
      run_one "${seed}" "${gpu}" "${RUN_ROOT}/seed${seed}" \
        >"${RUN_ROOT}/logs/seed${seed}.log" 2>&1 &
      pids+=("$!")
    done
    status=0
    for pid in "${pids[@]}"; do
      wait "${pid}" || status=1
    done
    (( status == 0 )) || exit "${status}"

    aggregate_args=("${RUN_ROOT}/three_seed_horizon_summary.json" --bootstrap-repetitions 10000 --seed 20260831)
    for seed in "${CONTROLLER_SEEDS[@]}"; do
      aggregate_args+=(--record "${seed}=${RUN_ROOT}/seed${seed}/online_full_controller_traces.jsonl")
    done
    "${PYTHON}" scripts/aggregate_online_full_controller.py "${aggregate_args[@]}" \
      >"${RUN_ROOT}/logs/aggregate.log" 2>&1
    ;;
  *)
    usage
    exit 2
    ;;
esac

echo "completed ${STAGE} under ${RUN_ROOT}"
