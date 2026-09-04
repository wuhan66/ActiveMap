#!/usr/bin/env bash
set -euo pipefail

RUN_ROOT="${1:?usage: queue_muno21_dpo_paired_assessment_nts.sh RUN_ROOT LABEL SEED}"
LABEL="${2:?usage: queue_muno21_dpo_paired_assessment_nts.sh RUN_ROOT LABEL SEED}"
SEED="${3:?usage: queue_muno21_dpo_paired_assessment_nts.sh RUN_ROOT LABEL SEED}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
PYTHON="${PYTHON:-/home/wh/venvs/activemap/bin/python}"
BASE=/mnt/mydisk/wh/ActiveMap/runs/agent/muno21_qwen3_4b_safety_dpo_v2_persistent_seed20260821/evaluation/sft-checkpoint-200-quality-v2
CANDIDATE=${RUN_ROOT}/evaluation/${LABEL}
SELECTION=${RUN_ROOT}/evaluation/selection

while [[ ! -f "${RUN_ROOT}/evaluation/COMPLETED" ]]; do
  echo "[$(date --iso-8601=seconds)] waiting for ${LABEL} validation"
  sleep 60
done

required=(
  "${BASE}/actions/predictions.jsonl"
  "${BASE}/rollouts/qwen3_4b_sft.jsonl"
  "${BASE}/writeback/writeback.jsonl"
  "${CANDIDATE}/actions/predictions.jsonl"
  "${CANDIDATE}/rollouts/qwen3_4b_sft.jsonl"
  "${CANDIDATE}/writeback/writeback.jsonl"
  "${SELECTION}/promotion_decision.json"
)
while true; do
  ready=true
  for path in "${required[@]}"; do
    [[ -s "${path}" ]] || ready=false
  done
  [[ "${ready}" == true ]] && break
  echo "[$(date --iso-8601=seconds)] waiting for paired-assessment inputs"
  sleep 60
done

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
"${PYTHON}" scripts/compare_agent_action_predictions.py \
  "${BASE}/actions/predictions.jsonl" "${CANDIDATE}/actions/predictions.jsonl" \
  "${SELECTION}/paired_action_bootstrap.json" --bootstrap 5000 --seed "${SEED}"
"${PYTHON}" scripts/compare_agent_rollouts.py \
  "${BASE}/rollouts/qwen3_4b_sft.jsonl" "${CANDIDATE}/rollouts/qwen3_4b_sft.jsonl" \
  "${SELECTION}/paired_rollout_bootstrap.json" --bootstrap 5000 --seed "${SEED}"
"${PYTHON}" scripts/compare_agent_writebacks.py \
  "${BASE}/writeback/writeback.jsonl" "${CANDIDATE}/writeback/writeback.jsonl" \
  "${SELECTION}/paired_writeback_bootstrap.json" --bootstrap 5000 --seed "${SEED}" \
  --group-key task_id --split val
"${PYTHON}" scripts/assess_agent_dpo_paired.py \
  "${SELECTION}/paired_action_bootstrap.json" \
  "${SELECTION}/paired_rollout_bootstrap.json" \
  "${SELECTION}/promotion_decision.json" \
  "${SELECTION}/paired_promotion_decision.json" \
  --writeback-comparison "${SELECTION}/paired_writeback_bootstrap.json"

date -Is >"${SELECTION}/PAIRED_COMPLETED"
echo "[$(date --iso-8601=seconds)] ${LABEL} paired assessment completed"
