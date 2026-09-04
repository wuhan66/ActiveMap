#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
SEED="${SEED:-20260730}"
LIMIT="${LIMIT:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-20260729}"
RUN_TAG="${RUN_TAG:-wiring_v2}"

STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_selector_v1/states_train_val_step0.jsonl"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
SELECTOR="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${SEED}/edit_utility_seed${SEED}/best.pt"
TOOL_BELIEF="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${SEED}/best_promoted.pt"
RUN_ROOT="${STORAGE_ROOT}/runs/sn7_active_catalog"

names=(noadapter base updated tools)
adapters=(
  ""
  "${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_base_seed${SEED}/best.pt"
  "${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_updated_seed${SEED}/best.pt"
  "${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_tools_seed${SEED}/best.pt"
)

pids=()
for index in "${!names[@]}"; do
  name="${names[$index]}"
  output="${RUN_ROOT}/sn7_step0_causal_${name}_n${LIMIT}_seed${SEED}_${RUN_TAG}"
  test ! -e "${output}" || {
    echo "refusing to overwrite ${output}" >&2
    exit 3
  }
  command=(
    "${PYTHON}"
    "${PROJECT_ROOT}/scripts/evaluate_active_catalog_closed_loop_baselines.py"
    "${STATES}"
    "${output}"
    --episodes "${EPISODES}"
    --learned-selector "edit_utility=${SELECTOR}"
    --policy edit_utility
    --seed "${SEED}"
    --device cuda:0
    --split val
    --tool-mode forced
    --belief-mode recurrent
    --tool-belief-checkpoint "${TOOL_BELIEF}"
    --tool-artifact-root "${output}/artifacts"
    --tool-out-size 256
    --max-candidates 16
    --max-acquisitions 2
    --bootstrap-repetitions 0
    --limit "${LIMIT}"
    --sample-seed "${SAMPLE_SEED}"
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  )
  if [[ -n "${adapters[$index]}" ]]; then
    command+=(--post-tool-action-adapter "${adapters[$index]}")
  fi
  CUDA_VISIBLE_DEVICES="${index}" \
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${command[@]}" \
    >"${STORAGE_ROOT}/logs/sn7_step0_causal_${name}_n${LIMIT}_seed${SEED}_${RUN_TAG}.log" \
    2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
