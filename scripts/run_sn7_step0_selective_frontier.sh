#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
LIMIT="${LIMIT:-6369}"
SAMPLE_SEED="${SAMPLE_SEED:-20260729}"
RUN_TAG="${RUN_TAG:-frontier_v1}"
SEEDS=(${SEEDS:-20260730 20260731 20260801})
RATES=(${RATES:-05 15 30})

STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_selector_v1/states_train_val_step0.jsonl"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog"
LOG_ROOT="${STORAGE_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

rate_value() {
  case "$1" in
    05) echo "0.05" ;;
    15) echo "0.15" ;;
    30) echo "0.30" ;;
    *) echo "unsupported rate label: $1" >&2; return 2 ;;
  esac
}

calibrate_one() {
  local seed="$1"
  local rate="$2"
  local gpu="$3"
  local selector="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${seed}/edit_utility_seed${seed}/best.pt"
  local gate_dir="${RUN_ROOT}/step0_selector_uncertainty_gate_seed${seed}_q${rate}_${RUN_TAG}"
  local log="${LOG_ROOT}/step0_selector_uncertainty_gate_seed${seed}_q${rate}_${RUN_TAG}.log"
  test -f "${selector}"
  if [[ -f "${gate_dir}/gate.json" && -f "${gate_dir}/summary.json" ]]; then
    echo "reuse calibrated gate ${gate_dir}"
    return 0
  fi
  test ! -e "${gate_dir}" || {
    echo "incomplete gate directory exists: ${gate_dir}" >&2
    return 3
  }
  CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/calibrate_selector_uncertainty_tool_gate.py" \
      "${STATES}" "${selector}" "${gate_dir}" \
      --target-call-rate "$(rate_value "${rate}")" \
      --device cuda:0 --batch-size 512 >"${log}" 2>&1
}

evaluate_one() {
  local seed="$1"
  local rate="$2"
  local gpu="$3"
  local selector="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${seed}/edit_utility_seed${seed}/best.pt"
  local tool_belief="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${seed}/best_promoted.pt"
  local adapter="${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_updated_seed${seed}/best.pt"
  local gate_dir="${RUN_ROOT}/step0_selector_uncertainty_gate_seed${seed}_q${rate}_${RUN_TAG}"
  local output="${RUN_ROOT}/sn7_step0_selective_updated_q${rate}_n${LIMIT}_seed${seed}_${RUN_TAG}"
  local log="${LOG_ROOT}/sn7_step0_selective_updated_q${rate}_n${LIMIT}_seed${seed}_${RUN_TAG}.log"
  test -f "${selector}"
  test -f "${tool_belief}"
  test -f "${adapter}"
  test -f "${gate_dir}/gate.json"
  test -f "${gate_dir}/summary.json"
  if [[ -f "${output}/summary.json" ]]; then
    echo "reuse completed evaluation ${output}"
    return 0
  fi
  test ! -e "${output}" || {
    echo "incomplete evaluation directory exists: ${output}" >&2
    return 3
  }
  CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_active_catalog_closed_loop_baselines.py" \
      "${STATES}" "${output}" \
      --episodes "${EPISODES}" \
      --learned-selector "edit_utility=${selector}" \
      --policy edit_utility \
      --seed "${seed}" --device cuda:0 --split val \
      --tool-mode selective --belief-mode recurrent \
      --tool-belief-checkpoint "${tool_belief}" \
      --post-tool-action-adapter "${adapter}" \
      --tool-gate "${gate_dir}/gate.json" \
      --tool-gate-summary "${gate_dir}/summary.json" \
      --tool-artifact-root "${output}/artifacts" \
      --tool-out-size 256 \
      --max-candidates 16 --max-acquisitions 2 \
      --bootstrap-repetitions 0 \
      --limit "${LIMIT}" --sample-seed "${SAMPLE_SEED}" \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >"${log}" 2>&1
}

run_batched() {
  local function_name="$1"
  shift
  local jobs=("$@")
  local index=0
  while (( index < ${#jobs[@]} )); do
    local pids=()
    local labels=()
    for gpu in 0 1 2 3 4; do
      (( index < ${#jobs[@]} )) || break
      IFS=: read -r seed rate <<<"${jobs[$index]}"
      "${function_name}" "${seed}" "${rate}" "${gpu}" &
      pids+=("$!")
      labels+=("${function_name}:${seed}:q${rate}:gpu${gpu}")
      ((index += 1))
    done
    local status=0
    for job_index in "${!pids[@]}"; do
      if wait "${pids[$job_index]}"; then
        echo "completed ${labels[$job_index]}"
      else
        echo "failed ${labels[$job_index]}" >&2
        status=1
      fi
    done
    (( status == 0 )) || exit "${status}"
  done
}

jobs=()
for seed in "${SEEDS[@]}"; do
  for rate in "${RATES[@]}"; do
    jobs+=("${seed}:${rate}")
  done
done

run_batched calibrate_one "${jobs[@]}"
run_batched evaluate_one "${jobs[@]}"
echo "completed SN7 Step-0 selective tool frontier"
