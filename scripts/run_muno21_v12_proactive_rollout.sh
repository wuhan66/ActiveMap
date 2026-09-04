#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_ENV:-${STORAGE_ROOT}/envs/activemap-agent}/bin/python"
GPU="${MUNO21_V12_GPU:-4}"
SEED="${MUNO21_AGENT_SEED:-20260821}"
SELECTOR_SEED="${MUNO21_SELECTOR_SEED:-20260811}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-4B}"
RUN_DIR="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed${SEED}"
GATE_DIR="${STORAGE_ROOT}/runs/agent/muno21_tool_need_gate_v11_seed20260831"
AGENT_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent"
OUTPUT="${MUNO21_V12_ROLLOUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v12_proactive_tool_val/seed${SEED}}"
METHODS="${MUNO21_V12_METHODS:-qwen3_4b_sft_calibrated_tool_to_belief,edit_conditioned_proactive_tools}"
ASSESS="${MUNO21_V12_ASSESS:-1}"
SPLIT="${MUNO21_V12_SPLIT:-val}"
TOOL_SUPERVISION="${MUNO21_V12_TOOL_SUPERVISION:-${AGENT_ROOT}/agent_data_v9_natural_sparse_tools/${SPLIT}/sft_composed.jsonl}"
TOOL_NEED_THRESHOLD="${MUNO21_V12_TOOL_NEED_THRESHOLD:-}"
ASSET_ROOT_MAP="${MUNO21_V12_ASSET_ROOT_MAP:-/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}}"
TOOL_BELIEF_CHECKPOINT="${MUNO21_V12_TOOL_BELIEF_CHECKPOINT:-${STORAGE_ROOT}/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt}"
V11="${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v11_calibrated_tool_val/seed${SEED}/summary.json"
FREEZE="${GATE_DIR}/controller_freeze.json"
ADAPTER_OVERRIDE="${MUNO21_V12_ADAPTER:-}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${ADAPTER_OVERRIDE}" ]]; then
  [[ -d "${ADAPTER_OVERRIDE}" ]] || {
    echo "missing adapter override: ${ADAPTER_OVERRIDE}" >&2
    exit 3
  }
  adapter="${ADAPTER_OVERRIDE}"
else
  adapter="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter_path"])' "${FREEZE}")"
fi
threshold_args=()
if [[ -n "${TOOL_NEED_THRESHOLD}" ]]; then
  threshold_args+=(--tool-need-threshold "${TOOL_NEED_THRESHOLD}")
fi

if [[ ! -s "${OUTPUT}/summary.json" ]]; then
  mkdir -p "$(dirname "${OUTPUT}")"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${AGENT_ROOT}/selector_states_v1.jsonl" "${OUTPUT}" \
    --adapter "${adapter}" \
    --checkpoint "${STORAGE_ROOT}/runs/selector/muno21_evidence_conservative_v5_seed${SELECTOR_SEED}/best.pt" \
    --episodes "${AGENT_ROOT}/episodes_train_val_v1.jsonl" \
    --asset-root-map "${ASSET_ROOT_MAP}" \
    --tool-belief-checkpoint "${TOOL_BELIEF_CHECKPOINT}" \
    --tool-artifact-root "${OUTPUT}/tool_artifacts" \
    --tool-supervision-jsonl "${TOOL_SUPERVISION}" \
    --tool-need-gate "${GATE_DIR}/gate.joblib" \
    "${threshold_args[@]}" \
    --max-tool-calls 2 --split "${SPLIT}" --budgets 1.5,3.0,4.5 \
    --device cuda --selector-device cpu --seed "${SEED}" \
    --methods "${METHODS}" \
    >"${OUTPUT}.log" 2>&1
fi

if [[ "${ASSESS}" == "1" ]]; then
  diagnostic="$(dirname "${OUTPUT}")/diagnostic_promotion.json"
  if [[ ! -s "${diagnostic}" ]]; then
    "${PYTHON}" scripts/assess_muno21_v12_proactive.py \
      "${V11}" "${OUTPUT}/summary.json" "${diagnostic}"
  fi
  echo "MUNO21 v12 proactive rollout complete: ${diagnostic}"
else
  echo "MUNO21 v12 rollout complete without promotion assessment: ${OUTPUT}"
fi
