#!/usr/bin/env bash
set -euo pipefail

: "${SN7_TEST_EPISODES:?test-only EpisodeRecord JSONL is required}"
: "${UPDATER_CHECKPOINT:?frozen updater checkpoint is required}"
: "${MODEL:?frozen Qwen3-VL model is required}"
: "${SEED_ADAPTERS:?comma-separated SEED=ADAPTER entries are required}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
PYTHON="${ACTIVEMAP_PYTHON:-python}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog_frozen_test}"
GPU="${GPU:-0}"
BUDGETS="${BUDGETS:-1,2,4,8}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
"${PYTHON}" -c 'from activemap.frozen_test import assert_frozen_test_access; assert_frozen_test_access()'
[[ ! -e "${RUN_ROOT}" ]] || { echo "refusing to reuse frozen test root: ${RUN_ROOT}" >&2; exit 41; }
pids="$(nvidia-smi -i "${GPU}" --query-compute-apps=pid --format=csv,noheader,nounits)"
[[ -z "${pids//[[:space:]]/}" ]] || { echo "GPU ${GPU} is occupied: ${pids}" >&2; exit 43; }
mkdir -p "${RUN_ROOT}"

"${PYTHON}" -m activemap build-selector-oracle \
  "${UPDATER_CHECKPOINT}" "${SN7_TEST_EPISODES}" "${RUN_ROOT}/selector_states_test.jsonl" \
  --device "cuda:${GPU}" --budgets "${BUDGETS}" --splits test --frozen-test
"${PYTHON}" scripts/prepare_active_catalog_closed_loop_bundle.py \
  "${RUN_ROOT}/selector_states_test.jsonl" "${SN7_TEST_EPISODES}" \
  "${RUN_ROOT}/bundle" --split test --frozen-test
"${PYTHON}" scripts/build_active_catalog_frozen_visual_index.py \
  "${RUN_ROOT}/bundle/states_test_step0.jsonl" "${RUN_ROOT}/bundle/episodes_test.jsonl" \
  "${RUN_ROOT}/visual_index"

IFS=',' read -ra ADAPTER_ROWS <<< "${SEED_ADAPTERS}"
record_args=()
no_tool_args=()
forced_args=()
selective_args=()
for row in "${ADAPTER_ROWS[@]}"; do
  seed="${row%%=*}"
  adapter="${row#*=}"
  [[ -n "${seed}" && -f "${adapter}/adapter_config.json" ]] || { echo "invalid seed adapter: ${row}" >&2; exit 42; }
  belief="${VALIDATION_RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}/joint_tool_belief_seed${seed}/best_promoted.pt"
  gate="${VALIDATION_RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}/joint_tool_gate_seed${seed}"
  for mode in none forced selective; do
    output="${RUN_ROOT}/seed${seed}/${mode}"
    extra=()
    if [[ "${mode}" != "none" ]]; then
      extra+=(--tool-mode "${mode}" --tool-belief-checkpoint "${belief}" --tool-artifact-root "${output}/tool_artifacts")
    fi
    if [[ "${mode}" == "selective" ]]; then
      extra+=(--tool-gate "${gate}/gate.joblib" --tool-gate-summary "${gate}/summary.json")
    fi
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
      "${MODEL}" "${adapter}" "${RUN_ROOT}/bundle/states_test_step0.jsonl" \
      "${RUN_ROOT}/bundle/episodes_test.jsonl" \
      "${RUN_ROOT}/visual_index/test_visual_prompts.jsonl" \
      "${RUN_ROOT}/visual_index/test_visual_index.jsonl" "${output}" \
      --gpu "${GPU}" --seed "${seed}" --split test --frozen-test \
      --max-candidates 16 --max-acquisitions 2 --max-new-tokens 64 \
      --bootstrap-repetitions 2000 "${extra[@]}"
    label="${mode}"
    [[ "${mode}" == "none" ]] && label="no_tool"
    record_args+=(--records "${seed}:${label}=${output}/evaluation/traces.jsonl")
    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${output}/evaluation/traces.jsonl" "${output}/writeback_input.jsonl" \
      --split test --frozen-test
    "${PYTHON}" scripts/launch_active_catalog_writeback.py \
      "${UPDATER_CHECKPOINT}" "${RUN_ROOT}/bundle/episodes_test.jsonl" \
      "${output}/writeback_input.jsonl" "${output}/writeback" --gpu "${GPU}" \
      --split test --frozen-test --image-size 512 --threshold 0.5
    case "${label}" in
      no_tool) no_tool_args+=(--baseline "${seed}=${output}/writeback/evaluation/writeback.jsonl") ;;
      forced) forced_args+=(--baseline "${seed}=${output}/writeback/evaluation/writeback.jsonl") ;;
      selective) selective_args+=(--candidate "${seed}=${output}/writeback/evaluation/writeback.jsonl") ;;
    esac
  done
done

"${PYTHON}" scripts/aggregate_active_catalog_tool_branches.py \
  "${RUN_ROOT}/branch_comparison.json" "${record_args[@]}" --split test --frozen-test
"${PYTHON}" scripts/assess_active_catalog_tool_branch_promotion.py \
  "${RUN_ROOT}/branch_comparison.json" "${RUN_ROOT}/branch_promotion.json" --min-seeds 3
"${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
  "${RUN_ROOT}/writeback_vs_no_tool.json" "${no_tool_args[@]}" "${selective_args[@]}" \
  --split test --frozen-test
"${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
  "${RUN_ROOT}/writeback_vs_forced.json" "${forced_args[@]}" "${selective_args[@]}" \
  --split test --frozen-test
"${PYTHON}" scripts/assess_active_catalog_tool_writeback_promotion.py \
  "${RUN_ROOT}/branch_promotion.json" "${RUN_ROOT}/writeback_vs_no_tool.json" \
  "${RUN_ROOT}/writeback_vs_forced.json" "${RUN_ROOT}/raw_promotion.json" --min-seeds 3 \
  --record-failed-upstream
