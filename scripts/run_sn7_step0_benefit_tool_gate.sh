#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
LIMIT="${LIMIT:-6369}"
SAMPLE_SEED="${SAMPLE_SEED:-20260729}"
RUN_TAG="${RUN_TAG:-benefit_gate_v1}"
SEEDS=(${SEEDS:-20260730 20260731 20260801})

STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_selector_v1/states_train_val_step0.jsonl"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
PAIRS="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_grounded_tool_pairs_seed30_v2"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog"
LOG_ROOT="${STORAGE_ROOT}/logs"

mkdir -p "${LOG_ROOT}"

run_seed() {
  local seed="$1"
  local gpu="$2"
  local selector="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${seed}/edit_utility_seed${seed}/best.pt"
  local tool_belief="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${seed}/best_promoted.pt"
  local adapter="${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_updated_seed${seed}/best.pt"
  local features="${RUN_ROOT}/step0_tool_gate_features_seed${seed}_${RUN_TAG}"
  local gate="${RUN_ROOT}/step0_tool_need_gate_seed${seed}_${RUN_TAG}"
  local output="${RUN_ROOT}/sn7_step0_selective_benefit_n${LIMIT}_seed${seed}_${RUN_TAG}"
  local log="${LOG_ROOT}/sn7_step0_selective_benefit_n${LIMIT}_seed${seed}_${RUN_TAG}.log"

  test -f "${selector}"
  test -f "${tool_belief}"
  test -f "${adapter}"

  for split in train val; do
    local split_features="${features}/${split}"
    if [[ ! -f "${split_features}/summary.json" ]]; then
      test ! -e "${split_features}" || {
        echo "incomplete feature directory exists: ${split_features}" >&2
        return 3
      }
      CUDA_VISIBLE_DEVICES="${gpu}" \
        PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
        "${PYTHON}" "${PROJECT_ROOT}/scripts/build_active_catalog_tool_gate_features.py" \
          "${PAIRS}/${split}/${split}.jsonl" \
          "${tool_belief}" "${split_features}" \
          --split "${split}" --device cuda:0 \
          >"${LOG_ROOT}/step0_tool_gate_features_seed${seed}_${split}_${RUN_TAG}.log" 2>&1
    fi
  done

  if [[ ! -f "${gate}/summary.json" ]]; then
    test ! -e "${gate}" || {
      echo "incomplete gate directory exists: ${gate}" >&2
      return 3
    }
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}/scripts" \
      "${PYTHON}" "${PROJECT_ROOT}/scripts/train_visual_tool_gate.py" \
        "${features}/train" "${features}/val" "${gate}" \
        --seed "${seed}" \
        --selection-objective proxy_utility \
        --fit-weighting utility_risk \
        --max-call-rate 0.15 \
        --min-oof-recall 0.10 \
        >"${LOG_ROOT}/step0_tool_need_gate_seed${seed}_${RUN_TAG}.log" 2>&1
  fi

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
      --tool-gate "${gate}/gate.joblib" \
      --tool-gate-summary "${gate}/summary.json" \
      --tool-artifact-root "${output}/artifacts" \
      --tool-out-size 256 \
      --max-candidates 16 --max-acquisitions 2 \
      --bootstrap-repetitions 0 \
      --limit "${LIMIT}" --sample-seed "${SAMPLE_SEED}" \
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" >>"${log}" 2>&1
}

pids=()
labels=()
gpu=0
for seed in "${SEEDS[@]}"; do
  run_seed "${seed}" "${gpu}" &
  pids+=("$!")
  labels+=("${seed}:gpu${gpu}")
  ((gpu += 1))
done

status=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "completed ${labels[$index]}"
  else
    echo "failed ${labels[$index]}" >&2
    status=1
  fi
done
exit "${status}"
