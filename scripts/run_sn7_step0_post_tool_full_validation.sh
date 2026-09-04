#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
LIMIT="${LIMIT:-6369}"
SAMPLE_SEED="${SAMPLE_SEED:-20260729}"
RUN_TAG="${RUN_TAG:-fullval_v1}"
SEEDS=(${SEEDS:-20260730 20260731 20260801})
VARIANTS=(notool noadapter base updated tools)

STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_selector_v1/states_train_val_step0.jsonl"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog"
LOG_ROOT="${STORAGE_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

run_one() {
  local seed="$1"
  local variant="$2"
  local gpu="$3"
  local selector="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${seed}/edit_utility_seed${seed}/best.pt"
  local tool_belief="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${seed}/best_promoted.pt"
  local output="${RUN_ROOT}/sn7_step0_causal_${variant}_n${LIMIT}_seed${seed}_${RUN_TAG}"
  local log="${LOG_ROOT}/sn7_step0_causal_${variant}_n${LIMIT}_seed${seed}_${RUN_TAG}.log"
  test -f "${selector}"
  test ! -e "${output}" || {
    echo "refusing to overwrite ${output}" >&2
    return 3
  }
  local command=(
    "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_active_catalog_closed_loop_baselines.py"
    "${STATES}" "${output}"
    --episodes "${EPISODES}"
    --learned-selector "edit_utility=${selector}"
    --policy edit_utility
    --seed "${seed}" --device cuda:0 --split val
    --belief-mode recurrent
    --max-candidates 16 --max-acquisitions 2
    --bootstrap-repetitions 0
    --limit "${LIMIT}" --sample-seed "${SAMPLE_SEED}"
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  )
  if [[ "${variant}" == notool ]]; then
    command+=(--tool-mode none)
  else
    test -f "${tool_belief}"
    command+=(
      --tool-mode forced
      --tool-belief-checkpoint "${tool_belief}"
      --tool-artifact-root "${output}/artifacts"
      --tool-out-size 256
    )
    if [[ "${variant}" != noadapter ]]; then
      local adapter="${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_${variant}_seed${seed}/best.pt"
      test -f "${adapter}"
      command+=(--post-tool-action-adapter "${adapter}")
    fi
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" \
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${command[@]}" >"${log}" 2>&1
}

jobs=()
for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    jobs+=("${seed}:${variant}")
  done
done

index=0
while (( index < ${#jobs[@]} )); do
  pids=()
  labels=()
  for gpu in 0 1 2 3 4; do
    (( index < ${#jobs[@]} )) || break
    IFS=: read -r seed variant <<<"${jobs[$index]}"
    run_one "${seed}" "${variant}" "${gpu}" &
    pids+=("$!")
    labels+=("${seed}:${variant}:gpu${gpu}")
    ((index += 1))
  done
  status=0
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

echo "completed SN7 Step-0 full validation matrix"
