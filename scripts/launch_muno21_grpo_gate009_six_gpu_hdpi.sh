#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
MATRIX=${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n512_matrix_v1
OUTPUT=${STORE}/runs/agent/muno21_grpo_t2p2_gate009_matrix_v1
REPORT=${STORE}/reports/muno21_grpo_t2p2_gate009_matrix_20260802
LOGROOT=${STORE}/logs/muno21_grpo_t2p2_gate009_matrix_v1
BASELINE=${STORE}/artifacts/paper_rollouts/agent_v12_proactive_tool_four_seed_threshold009_v1/seed20260822/qwen3_4b_sft_calibrated_tool_to_belief.jsonl
mkdir -p "${OUTPUT}" "${REPORT}" "${LOGROOT}"
cd "${PROJECT}"

run_one() {
  local gpu=$1
  local variant=$2
  local seed=$3
  local adapter=${MATRIX}/${variant}_seed${seed}/final
  local run=${OUTPUT}/${variant}_seed${seed}
  MUNO21_V12_GPU="${gpu}" \
  MUNO21_AGENT_SEED="${seed}" \
  MUNO21_SELECTOR_SEED=20260811 \
  MUNO21_V12_ROLLOUT_ROOT="${run}" \
  MUNO21_V12_METHODS=qwen3_4b_sft_calibrated_tool_to_belief \
  MUNO21_V12_TOOL_NEED_THRESHOLD=0.09 \
  MUNO21_V12_ADAPTER="${adapter}" \
  MUNO21_V12_ASSESS=0 \
  bash scripts/run_muno21_v12_proactive_rollout.sh \
    >"${LOGROOT}/${variant}_seed${seed}.log" 2>&1 &
  echo "$! GPU${gpu} ${variant} seed${seed} gate0.09"
}

run_one 2 constrained 20260971
run_one 3 constrained 20260972
run_one 4 constrained 20260973
run_one 5 unconstrained 20260974
run_one 6 unconstrained 20260975
run_one 7 unconstrained 20260976
wait

PYTHONPATH=src:. /home/wh/ActiveMap/envs/activemap-agent/bin/python scripts/summarize_muno21_grpo_v2.py "${REPORT}" \
  --baseline "${BASELINE}" \
  --run constrained_gate009=20260971=${OUTPUT}/constrained_seed20260971/qwen3_4b_sft_calibrated_tool_to_belief.jsonl \
  --run constrained_gate009=20260972=${OUTPUT}/constrained_seed20260972/qwen3_4b_sft_calibrated_tool_to_belief.jsonl \
  --run constrained_gate009=20260973=${OUTPUT}/constrained_seed20260973/qwen3_4b_sft_calibrated_tool_to_belief.jsonl \
  --run unconstrained_gate009=20260974=${OUTPUT}/unconstrained_seed20260974/qwen3_4b_sft_calibrated_tool_to_belief.jsonl \
  --run unconstrained_gate009=20260975=${OUTPUT}/unconstrained_seed20260975/qwen3_4b_sft_calibrated_tool_to_belief.jsonl \
  --run unconstrained_gate009=20260976=${OUTPUT}/unconstrained_seed20260976/qwen3_4b_sft_calibrated_tool_to_belief.jsonl
date -Is >"${REPORT}/PIPELINE_COMPLETED"
