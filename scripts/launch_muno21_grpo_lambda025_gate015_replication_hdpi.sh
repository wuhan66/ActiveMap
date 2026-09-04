#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
MATRIX=${STORE}/runs/agent/muno21_grpo_t2p2_seeded_hash_g8_n512_matrix_v1
OUTPUT=${STORE}/runs/agent/muno21_grpo_lambda025_gate_sweep_v1
LOGROOT=${STORE}/logs/muno21_grpo_lambda025_gate_sweep_v1
mkdir -p "${OUTPUT}" "${LOGROOT}"
cd "${PROJECT}"

run_one() {
  local gpu=$1 seed=$2
  local run=${OUTPUT}/gate015_seed${seed}
  local adapter=${MATRIX}/lambda025_seed${seed}/final
  if [[ -s "${run}/summary.json" ]]; then
    echo "skip_completed=gate015_seed${seed}"
    return
  fi
  [[ ! -e "${run}" ]] || { echo "Refusing incomplete output: ${run}" >&2; return 2; }
  MUNO21_V12_GPU="${gpu}" MUNO21_AGENT_SEED="${seed}" \
  MUNO21_SELECTOR_SEED=20260811 MUNO21_V12_ROLLOUT_ROOT="${run}" \
  MUNO21_V12_METHODS=qwen3_4b_sft_calibrated_tool_to_belief \
  MUNO21_V12_TOOL_NEED_THRESHOLD=0.15 MUNO21_V12_ADAPTER="${adapter}" \
  MUNO21_V12_ASSESS=0 bash scripts/run_muno21_v12_proactive_rollout.sh \
    >"${LOGROOT}/gate015_seed${seed}.log" 2>&1 &
  echo "$! GPU${gpu} gate015 seed${seed}"
}

run_one 0 20260979
run_one 1 20260980
wait
date -Is >"${OUTPUT}/GATE015_REPLICATION_COMPLETED"
