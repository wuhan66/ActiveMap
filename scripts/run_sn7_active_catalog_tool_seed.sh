#!/usr/bin/env bash
set -euo pipefail

: "${SEED:?set SEED to the model-training seed}"
: "${CLOSED_LOOP_ADAPTER:?set CLOSED_LOOP_ADAPTER to the seed-specific promoted adapter}"

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
RUN_ROOT="${RUN_ROOT:-/home/wh/ActiveMap/runs/sn7_active_catalog}"
export SEEDS="${SEED}"
export ACTIVE_SEED="${SEED}"
primary_seed="${ACTIVE_CATALOG_PRIMARY_SEED:-20260717}"
if [[ -z "${SEED_TAG:-}" ]]; then
  if [[ "${SEED}" == "${primary_seed}" ]]; then
    SEED_TAG="seed1"
  else
    SEED_TAG="seed${SEED}"
  fi
fi
export SEED_TAG
branch_suffix="_${SEED_TAG}"
[[ "${SEED_TAG}" == "seed1" ]] && branch_suffix=""

cd "${PROJECT_ROOT}"

run_stage() {
  echo "stage=$1 seed=${SEED} tag=${SEED_TAG}"
  bash scripts/run_sn7_active_catalog_qwen.sh "$1"
}

run_stage_if_missing() {
  local artifact="$1"
  local stage="$2"
  if [[ -s "${artifact}" ]]; then
    echo "skip=${stage} existing=${artifact}"
  else
    run_stage "${stage}"
  fi
}

run_stage_for_artifact_set() {
  local stage="$1"
  shift
  local present=0
  local artifact
  for artifact in "$@"; do
    [[ -s "${artifact}" ]] && present=$((present + 1))
  done
  if [[ "${present}" -eq "$#" ]]; then
    echo "skip=${stage} complete_artifact_set=$#"
  elif [[ "${present}" -gt 0 ]]; then
    echo "partial artifact set for ${stage}; refusing ambiguous resume" >&2
    return 2
  else
    run_stage "${stage}"
  fi
}

run_stage_if_missing "${RUN_ROOT}/closed_loop_${SEED_TAG}/evaluation/traces.jsonl" closed_loop_seed
run_stage_if_missing "${RUN_ROOT}/joint_transition_train_${SEED_TAG}/evaluation/joint_transitions.jsonl" joint_transition_train
run_stage_if_missing "${RUN_ROOT}/joint_transition_audit_${SEED_TAG}.json" audit_joint_transitions
run_stage_for_artifact_set ground_joint_tools \
  "${RUN_ROOT}/joint_grounded_tools${branch_suffix}/train/train.jsonl" \
  "${RUN_ROOT}/joint_grounded_tools${branch_suffix}/val/val.jsonl"
run_stage_if_missing "${RUN_ROOT}/joint_grounded_tools${branch_suffix}/audit.json" audit_grounded_joint_tools
run_stage_if_missing "${RUN_ROOT}/joint_tool_belief_${SEED_TAG}/summary.json" joint_tool_belief_seed
run_stage_if_missing "${RUN_ROOT}/joint_tool_belief_gated_${SEED_TAG}/summary.json" joint_tool_belief_gated_seed
run_stage_if_missing "${RUN_ROOT}/joint_tool_gate_features_${SEED_TAG}/val/summary.json" build_joint_tool_gate_features
run_stage_if_missing "${RUN_ROOT}/joint_tool_gate_${SEED_TAG}/summary.json" train_joint_tool_gate
run_stage_if_missing "${RUN_ROOT}/closed_loop_forced_tools_${SEED_TAG}/evaluation/traces.jsonl" closed_loop_forced_tools
run_stage_if_missing "${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/evaluation/traces.jsonl" closed_loop_selective_tools
gated_trace="${RUN_ROOT}/closed_loop_selective_tools_joint_tool_belief_gated_${SEED_TAG}/evaluation/traces.jsonl"
if [[ -s "${gated_trace}" ]]; then
  echo "skip=closed_loop_selective_tools reliability_gate=1 existing=${gated_trace}"
else
  echo "stage=closed_loop_selective_tools reliability_gate=1 seed=${SEED} tag=${SEED_TAG}"
  TOOL_BELIEF_FAMILY=joint_tool_belief_gated \
    bash scripts/run_sn7_active_catalog_qwen.sh closed_loop_selective_tools
fi
run_stage_if_missing "${RUN_ROOT}/closed_loop_reliability_gate_${SEED_TAG}.json" compare_closed_loop_reliability_gate
run_stage_if_missing "${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/paired_tool_branches.json" compare_closed_loop_tool_branches
run_stage_if_missing "${RUN_ROOT}/closed_loop_selective_tools_${SEED_TAG}/promotion.json" assess_closed_loop_tool_branches
run_stage_for_artifact_set prepare_tool_branch_writebacks \
  "${RUN_ROOT}/closed_loop_writeback_inputs/no_tool${branch_suffix}.jsonl" \
  "${RUN_ROOT}/closed_loop_writeback_inputs/forced_tools${branch_suffix}.jsonl" \
  "${RUN_ROOT}/closed_loop_writeback_inputs/selective_tools${branch_suffix}.jsonl"
run_stage_if_missing "${RUN_ROOT}/closed_loop_writeback_no_tool${branch_suffix}/evaluation/summary.json" closed_loop_writeback_no_tool
run_stage_if_missing "${RUN_ROOT}/closed_loop_writeback_forced_tools${branch_suffix}/evaluation/summary.json" closed_loop_writeback_forced_tools
run_stage_if_missing "${RUN_ROOT}/closed_loop_writeback_selective_tools${branch_suffix}/evaluation/summary.json" closed_loop_writeback_selective_tools
run_stage_for_artifact_set compare_tool_branch_writebacks \
  "${RUN_ROOT}/closed_loop_writeback_selective_tools${branch_suffix}/paired_no_tool_aoi.json" \
  "${RUN_ROOT}/closed_loop_writeback_selective_tools${branch_suffix}/paired_forced_tools_aoi.json"
run_stage_if_missing "${RUN_ROOT}/closed_loop_writeback_selective_tools${branch_suffix}/promotion.json" assess_tool_writeback_promotion
