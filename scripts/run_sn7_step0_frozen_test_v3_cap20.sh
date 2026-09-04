#!/usr/bin/env bash
set -euo pipefail

: "${SN7_TEST_MANIFEST:?path-only frozen test manifest is required}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${PYTHON:-${STORAGE_ROOT}/envs/activemap-agent/bin/python}"
REGISTRY="${REGISTRY:-${PROJECT_ROOT}/configs/experiments/sn7_step0_frozen_registry_v2.yaml}"
UPDATER="${UPDATER:-${STORAGE_ROOT}/models/frozen_updater/sn7_v4_hierarchical_vector_change_seed20260716/best_quality.pt}"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/sn7_active_catalog/step0_frozen_test_v3_cap20}"
LOG_ROOT="${LOG_ROOT:-${STORAGE_ROOT}/logs/sn7_step0_frozen_test_v3_cap20}"
BUDGETS="${BUDGETS:-1.5,3.0,4.5}"
SEEDS=(${SEEDS:-20260730 20260731 20260801})
GPUS=(${GPUS:-0 1 2})

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" -c \
  'from activemap.frozen_test import assert_frozen_test_access; assert_frozen_test_access()'
test -f "${REGISTRY}"
test -f "${UPDATER}"
test -f "${SN7_TEST_MANIFEST}"
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "refusing to reuse frozen test root: ${RUN_ROOT}" >&2
  exit 41
}
(( ${#GPUS[@]} >= ${#SEEDS[@]} )) || {
  echo "one GPU per seed is required" >&2
  exit 42
}
mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

SN7_TEST_EPISODES="${RUN_ROOT}/episodes_test.jsonl"
STATES="${RUN_ROOT}/states_test_step0.jsonl"
"${PYTHON}" -m activemap build-episodes-sn7 \
  "${SN7_TEST_MANIFEST}" "${SN7_TEST_EPISODES}" \
  --max-month-gap 1 --min-change-persistence 2 --min-area 16 \
  --max-per-operation 20 \
  --max-centroid-distance 20 --seed 20260710 --frozen-test \
  >"${LOG_ROOT}/build_episodes.log" 2>&1
"${PYTHON}" -m activemap audit-episodes \
  "${SN7_TEST_EPISODES}" "${RUN_ROOT}/episodes_test.audit.json" \
  --expected-derivation-version sn7-adjacent-v3-distance-gated \
  >>"${LOG_ROOT}/build_episodes.log" 2>&1
CUDA_VISIBLE_DEVICES="${GPUS[0]}" \
  "${PYTHON}" -m activemap build-selector-oracle \
    "${UPDATER}" "${SN7_TEST_EPISODES}" "${STATES}" \
    --device cuda:0 \
    --image-size 512 \
    --budgets "${BUDGETS}" \
    --splits test \
    --utility-mode executable \
    --utility-profile balanced \
    --writeback-delta-margin 0.15 \
    --asset-root-map /mnt/mydisk/wh/ActiveMap=/home/wh/ActiveMap \
    --frozen-test \
    >"${LOG_ROOT}/build_states.log" 2>&1

run_seed() {
  local seed="$1"
  local gpu="$2"
  local selector="${STORAGE_ROOT}/runs/selector/sn7_step0_two_stage_v2_seed${seed}/edit_utility_seed${seed}/best.pt"
  local belief="${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed${seed}/best_promoted.pt"
  local adapter="${STORAGE_ROOT}/runs/agent/sn7_step0_post_tool_adapter_updated_seed${seed}/best.pt"
  local gate_root="${STORAGE_ROOT}/runs/sn7_active_catalog/step0_tool_need_gate_seed${seed}_benefit_gate_v1"
  local gate="${gate_root}/gate.joblib"
  local gate_summary="${gate_root}/summary.json"
  for path in "${selector}" "${belief}" "${adapter}" "${gate}" "${gate_summary}"; do
    test -f "${path}"
  done

  for variant in notool forced benefit; do
    local root="${RUN_ROOT}/seed${seed}/${variant}"
    local eval_root="${root}/evaluation"
    local tool_args=()
    if [[ "${variant}" == "forced" ]]; then
      tool_args=(
        --tool-mode forced --belief-mode recurrent
        --tool-belief-checkpoint "${belief}"
        --post-tool-action-adapter "${adapter}"
        --tool-artifact-root "${root}/tool_artifacts"
      )
    elif [[ "${variant}" == "benefit" ]]; then
      tool_args=(
        --tool-mode selective --belief-mode recurrent
        --tool-belief-checkpoint "${belief}"
        --post-tool-action-adapter "${adapter}"
        --tool-gate "${gate}" --tool-gate-summary "${gate_summary}"
        --tool-artifact-root "${root}/tool_artifacts"
      )
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON}" scripts/evaluate_active_catalog_closed_loop_baselines.py \
        "${STATES}" "${eval_root}" \
        --episodes "${SN7_TEST_EPISODES}" \
        --learned-selector "edit_utility=${selector}" --policy edit_utility \
        --seed "${seed}" --device cuda:0 --split test --frozen-test \
        --max-candidates 16 --max-acquisitions 2 --tool-out-size 256 \
        --bootstrap-repetitions 0 \
        --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
        "${tool_args[@]}" >"${LOG_ROOT}/seed${seed}_${variant}_evaluation.log" 2>&1

    "${PYTHON}" scripts/convert_active_catalog_closed_loop_for_writeback.py \
      "${eval_root}/edit_utility.jsonl" "${root}/writeback_input.jsonl" \
      --split test --evidence-mode last --frozen-test \
      >"${LOG_ROOT}/seed${seed}_${variant}_conversion.log" 2>&1
    CUDA_VISIBLE_DEVICES="${gpu}" \
      "${PYTHON}" scripts/launch_active_catalog_writeback.py \
        "${UPDATER}" "${SN7_TEST_EPISODES}" "${root}/writeback_input.jsonl" \
        "${root}/writeback" --gpu 0 --python "${PYTHON}" \
        --image-size 512 --threshold 0.5 --delta-margin 0.15 \
        --evidence-fusion confidence_weighted \
        --protocol-name sn7-step0-frozen-test-v3-cap20 \
        --split test --frozen-test \
        --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
        --monitor-interval 5 >"${LOG_ROOT}/seed${seed}_${variant}_writeback.log" 2>&1
  done
}

pids=()
for index in "${!SEEDS[@]}"; do
  run_seed "${SEEDS[$index]}" "${GPUS[$index]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
(( status == 0 )) || exit "${status}"

summary_args=()
notool_writebacks=()
forced_writebacks=()
benefit_writebacks=()
for seed in "${SEEDS[@]}"; do
  summary_args+=(
    --notool "${seed}=${RUN_ROOT}/seed${seed}/notool/evaluation/edit_utility.jsonl"
    --forced "${seed}=${RUN_ROOT}/seed${seed}/forced/evaluation/edit_utility.jsonl"
    --benefit "${seed}=${RUN_ROOT}/seed${seed}/benefit/evaluation/edit_utility.jsonl"
  )
  notool_writebacks+=(--baseline "${seed}=${RUN_ROOT}/seed${seed}/notool/writeback/evaluation/writeback.jsonl")
  forced_writebacks+=(--baseline "${seed}=${RUN_ROOT}/seed${seed}/forced/writeback/evaluation/writeback.jsonl")
  benefit_writebacks+=(--candidate "${seed}=${RUN_ROOT}/seed${seed}/benefit/writeback/evaluation/writeback.jsonl")
done

"${PYTHON}" scripts/summarize_sn7_step0_three_policy.py \
  "${RUN_ROOT}/three_policy_summary.json" "${summary_args[@]}" \
  --split test --frozen-test --bootstrap-repetitions 10000 --bootstrap-seed 20260729
"${PYTHON}" scripts/assess_sn7_step0_frontier_promotion.py \
  "${RUN_ROOT}/three_policy_summary.json" "${RUN_ROOT}/frontier_promotion.json" \
  --min-seeds 3 --expected-split test
"${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
  "${RUN_ROOT}/benefit_vs_notool.json" \
  "${notool_writebacks[@]}" "${benefit_writebacks[@]}" \
  --split test --frozen-test --repetitions 10000 --seed 20260729
"${PYTHON}" scripts/aggregate_active_catalog_tool_writebacks.py \
  "${RUN_ROOT}/benefit_vs_forced.json" \
  "${forced_writebacks[@]}" "${benefit_writebacks[@]}" \
  --split test --frozen-test --repetitions 10000 --seed 20260729
"${PYTHON}" scripts/assess_active_catalog_tool_writeback_promotion.py \
  "${RUN_ROOT}/frontier_promotion.json" \
  "${RUN_ROOT}/benefit_vs_notool.json" "${RUN_ROOT}/benefit_vs_forced.json" \
  "${RUN_ROOT}/writeback_promotion.json" --min-seeds 3

"${PYTHON}" -c \
  'import json,sys; from pathlib import Path; root=Path(sys.argv[1]); promotion=json.loads((root/"writeback_promotion.json").read_text()); (root/"COMPLETE.json").write_text(json.dumps({"schema_version":"sn7-step0-frozen-test-complete-v2","promotion_passed":bool(promotion["promote"]),"test_assets_read":True},indent=2)+"\n")' \
  "${RUN_ROOT}"
echo "completed SN7 Step-0 frozen test v3 cap20"
