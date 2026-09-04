#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:?set SEED}"
GPU="${GPU:?set GPU}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
LIMIT="${LIMIT:-512}"
SAMPLE_SEED="${SAMPLE_SEED:-20260729}"

PAIR_ROOT="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_grounded_tool_pairs_seed30_v2"
TRAIN_PAIRS="${PAIR_ROOT}/train/train.jsonl"
VAL_PAIRS="${PAIR_ROOT}/val/val.jsonl"
STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/step0_selector_v1/states_train_val_step0.jsonl"
EPISODES="${STORAGE_ROOT}/processed/sn7_v1/agent/executable_selector_v3_512_sharded/closed_loop_val_bundle_v1/episodes_val.jsonl"
SELECTOR="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${SEED}/edit_utility_seed${SEED}/best.pt"
TOOL_BELIEF_DIR="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${SEED}"
LOG_ROOT="${STORAGE_ROOT}/logs"

mkdir -p "${LOG_ROOT}"
for path in "${TRAIN_PAIRS}" "${VAL_PAIRS}" "${STATES}" "${EPISODES}" "${SELECTOR}"; do
  test -f "${path}" || {
    echo "missing input: ${path}" >&2
    exit 2
  }
done

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
  "${PYTHON}" "${PROJECT_ROOT}/scripts/train_post_acquisition_tool_belief.py" \
  "${TRAIN_PAIRS}" "${VAL_PAIRS}" "${TOOL_BELIEF_DIR}" \
  --device cuda:0 --epochs 120 --batch-size 128 --patience 20 \
  --seed "${SEED}" --reliability-gate --gate-bias -1.5 \
  >"${LOG_ROOT}/sn7_step0_tool_belief_gated_seed${SEED}.log" 2>&1

TOOL_BELIEF="${TOOL_BELIEF_DIR}/best_promoted.pt"
test -f "${TOOL_BELIEF}" || TOOL_BELIEF="${TOOL_BELIEF_DIR}/best.pt"

adapter_names=(base updated tools)
adapter_flags=("--use-post-acquisition-belief" "" "--include-tool-results")
for index in "${!adapter_names[@]}"; do
  name="${adapter_names[$index]}"
  output="${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_${name}_seed${SEED}"
  command=(
    "${PYTHON}" "${PROJECT_ROOT}/scripts/train_post_tool_action_adapter.py"
    "${TRAIN_PAIRS}" "${VAL_PAIRS}" "${TOOL_BELIEF}" "${output}"
    --device cuda:0 --seed "${SEED}" --epochs 80 --batch-size 128 --patience 12
  )
  if [[ -n "${adapter_flags[$index]}" ]]; then
    command+=("${adapter_flags[$index]}")
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${command[@]}" \
    >"${LOG_ROOT}/sn7_step0_post_tool_adapter_${name}_seed${SEED}.log" 2>&1
done

for name in noadapter base updated tools; do
  output="${STORAGE_ROOT}/runs/sn7_active_catalog/sn7_step0_causal_${name}_n${LIMIT}_seed${SEED}_wiring_v2"
  command=(
    "${PYTHON}" "${PROJECT_ROOT}/scripts/evaluate_active_catalog_closed_loop_baselines.py"
    "${STATES}" "${output}"
    --episodes "${EPISODES}"
    --learned-selector "edit_utility=${SELECTOR}"
    --policy edit_utility
    --seed "${SEED}" --device cuda:0 --split val
    --tool-mode forced --belief-mode recurrent
    --tool-belief-checkpoint "${TOOL_BELIEF}"
    --tool-artifact-root "${output}/artifacts"
    --tool-out-size 256 --max-candidates 16 --max-acquisitions 2
    --bootstrap-repetitions 0 --limit "${LIMIT}" --sample-seed "${SAMPLE_SEED}"
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}"
  )
  if [[ "${name}" != noadapter ]]; then
    command+=(
      --post-tool-action-adapter
      "${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_${name}_seed${SEED}/best.pt"
    )
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${command[@]}" \
    >"${LOG_ROOT}/sn7_step0_causal_${name}_n${LIMIT}_seed${SEED}_wiring_v2.log" \
    2>&1
done

echo "completed SN7 Step-0 Tool-Belief causal pipeline seed=${SEED}"
