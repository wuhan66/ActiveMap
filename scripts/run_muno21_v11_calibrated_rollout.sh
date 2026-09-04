#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${ACTIVEMAP_PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_ENV:-${STORAGE_ROOT}/envs/activemap-agent}/bin/python"
GPU="${MUNO21_V11_GPU:-4}"
SEED="${MUNO21_AGENT_SEED:-20260821}"
SELECTOR_SEED="${MUNO21_SELECTOR_SEED:-20260811}"
MODEL="${MUNO21_AGENT_MODEL:-/home/wh/hf_models/Qwen3-4B}"
RUN_DIR="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_balanced_tool_sft_seed${SEED}"
GATE_DIR="${STORAGE_ROOT}/runs/agent/muno21_tool_need_gate_v11_seed20260831"
AGENT_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent"
OUTPUT="${MUNO21_V11_ROLLOUT_ROOT:-${STORAGE_ROOT}/artifacts/paper_rollouts/agent_v11_calibrated_tool_val/seed${SEED}}"
FREEZE="${GATE_DIR}/controller_freeze.json"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ ! -s "${FREEZE}" ]]; then
  "${PYTHON}" scripts/select_muno21_v11_controller.py \
    "${RUN_DIR}/evaluation/selection/static_checkpoint_decision.json" \
    "${GATE_DIR}/summary.json" "${RUN_DIR}" "${FREEZE}"
fi
adapter="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["adapter_path"])' "${FREEZE}")"

if [[ -s "${OUTPUT}/summary.json" ]]; then
  echo "MUNO21 v11 rollout already complete: ${OUTPUT}"
else
  mkdir -p "$(dirname "${OUTPUT}")"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" scripts/evaluate_agent_rollouts.py \
    "${MODEL}" "${AGENT_ROOT}/selector_states_v1.jsonl" "${OUTPUT}" \
    --adapter "${adapter}" \
    --checkpoint "${STORAGE_ROOT}/runs/selector/muno21_evidence_conservative_v5_seed${SELECTOR_SEED}/best.pt" \
    --generic-checkpoint "${STORAGE_ROOT}/runs/selector/muno21_evidence_generic_v5_seed${SELECTOR_SEED}/best.pt" \
    --episodes "${AGENT_ROOT}/episodes_train_val_v1.jsonl" \
    --tool-belief-checkpoint "${STORAGE_ROOT}/runs/agent/tool_belief_anchored_v4_seed20260821/best.pt" \
    --tool-artifact-root "${OUTPUT}/tool_artifacts" \
    --tool-supervision-jsonl "${AGENT_ROOT}/agent_data_v9_natural_sparse_tools/val/sft_composed.jsonl" \
    --tool-need-gate "${GATE_DIR}/gate.joblib" \
    --max-tool-calls 2 --split val --budgets 1.5,3.0,4.5 \
    --device cuda --selector-device cpu --seed "${SEED}" \
    --methods oracle,generic_selector,edit_conditioned_selector,forced_tools,qwen3_4b_sft_tools_no_belief,qwen3_4b_sft_tool_to_belief,qwen3_4b_sft_calibrated_tool_to_belief \
    >"${OUTPUT}.log" 2>&1
fi

diagnostic="$(dirname "${OUTPUT}")/diagnostic_promotion.json"
if [[ ! -s "${diagnostic}" ]]; then
  "${PYTHON}" scripts/assess_muno21_v11_diagnostic.py \
    "${OUTPUT}/summary.json" "${diagnostic}"
fi
echo "MUNO21 v11 calibrated rollout complete: ${diagnostic}"
