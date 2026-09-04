#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
CHECKPOINT="${CHECKPOINT:-${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt}"
EPISODES="${EPISODES:-${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/step0_validation_writeback_v1}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs}"
SEEDS=(${SEEDS:-20260730 20260731 20260801})
VARIANTS=(notool forced benefit)

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"
test -f "${CHECKPOINT}"
test -f "${EPISODES}"

trace_path() {
  local seed="$1"
  local variant="$2"
  case "${variant}" in
    notool)
      echo "${STORAGE_ROOT}/runs/sn7_active_catalog/sn7_step0_causal_notool_n6369_seed${seed}_fullval_v1/edit_utility.jsonl"
      ;;
    forced)
      echo "${STORAGE_ROOT}/runs/sn7_active_catalog/sn7_step0_causal_updated_n6369_seed${seed}_fullval_v1/edit_utility.jsonl"
      ;;
    benefit)
      echo "${STORAGE_ROOT}/runs/sn7_active_catalog/sn7_step0_selective_benefit_n6369_seed${seed}_benefit_gate_v1/edit_utility.jsonl"
      ;;
    *)
      echo "unsupported variant: ${variant}" >&2
      return 2
      ;;
  esac
}

prepare_input() {
  local seed="$1"
  local variant="$2"
  local root="${RUN_ROOT}/seed${seed}/${variant}"
  local input="${root}/writeback_input.jsonl"
  local trace
  trace="$(trace_path "${seed}" "${variant}")"
  test -f "${trace}"
  mkdir -p "${root}"
  if [[ ! -f "${input}" ]]; then
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
      "${PYTHON}" "${PROJECT_ROOT}/scripts/convert_active_catalog_closed_loop_for_writeback.py" \
        "${trace}" "${input}" --split val --evidence-mode last
  fi
}

run_one() {
  local seed="$1"
  local variant="$2"
  local gpu="$3"
  local root="${RUN_ROOT}/seed${seed}/${variant}"
  local output="${root}/writeback"
  local log="${LOG_ROOT}/sn7_step0_writeback_seed${seed}_${variant}_v1.log"
  if [[ -f "${output}/process_result.json" ]]; then
    echo "reuse completed writeback ${output}"
    return 0
  fi
  test ! -e "${output}" || {
    echo "incomplete writeback directory exists: ${output}" >&2
    return 3
  }
  cd "${PROJECT_ROOT}"
  PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${CHECKPOINT}" "${EPISODES}" "${root}/writeback_input.jsonl" "${output}" \
      --gpu "${gpu}" --python "${PYTHON}" \
      --image-size 512 --threshold 0.5 --delta-margin 0.15 \
      --evidence-fusion confidence_weighted \
      --protocol-name sn7-step0-benefit-validation-writeback-v1 \
      --split val \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
      --monitor-interval 5 >"${log}" 2>&1
}

jobs=()
for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    prepare_input "${seed}" "${variant}"
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

echo "completed SN7 Step-0 validation writeback matrix"
