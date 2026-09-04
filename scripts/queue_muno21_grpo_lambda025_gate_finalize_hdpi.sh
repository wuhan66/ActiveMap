#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/wh/projects/activemap-v1
STORE=/home/wh/ActiveMap
PY=${STORE}/envs/activemap-agent/bin/python
ROOT=${STORE}/runs/agent/muno21_grpo_lambda025_gate_sweep_v1
REPORT=${STORE}/reports/muno21_grpo_lambda025_gate_sweep_20260802
BASELINE=${STORE}/artifacts/paper_rollouts/agent_v12_proactive_tool_four_seed_threshold009_v1/seed20260822/qwen3_4b_sft_calibrated_tool_to_belief.jsonl
METHOD=qwen3_4b_sft_calibrated_tool_to_belief

cd "${PROJECT}"
expected=(
  gate005_seed20260977 gate005_seed20260979 gate005_seed20260980
  gate007_seed20260977 gate007_seed20260979 gate007_seed20260980
  gate008_seed20260977 gate008_seed20260979 gate008_seed20260980
  gate009_seed20260977 gate009_seed20260979 gate009_seed20260980
  gate010_seed20260977 gate010_seed20260979 gate010_seed20260980
  gate011_seed20260977 gate011_seed20260979 gate011_seed20260980
  gate013_seed20260977 gate013_seed20260979 gate013_seed20260980
  gate015_seed20260977 gate015_seed20260979 gate015_seed20260980
)
while true; do
  ready=0
  for run in "${expected[@]}"; do
    [[ -s "${ROOT}/${run}/${METHOD}.jsonl" ]] && ready=$((ready + 1))
  done
  echo "$(date -Is) ready=${ready}/${#expected[@]}"
  [[ ${ready} -eq ${#expected[@]} ]] && break
  sleep 60
done

mkdir -p "${REPORT}"
PYTHONPATH=src:. "${PY}" scripts/summarize_gate_threshold_sweep.py "${REPORT}" \
  --baseline "${BASELINE}" \
  --run 0.05=20260977="${ROOT}/gate005_seed20260977/${METHOD}.jsonl" \
  --run 0.05=20260979="${ROOT}/gate005_seed20260979/${METHOD}.jsonl" \
  --run 0.05=20260980="${ROOT}/gate005_seed20260980/${METHOD}.jsonl" \
  --run 0.07=20260977="${ROOT}/gate007_seed20260977/${METHOD}.jsonl" \
  --run 0.07=20260979="${ROOT}/gate007_seed20260979/${METHOD}.jsonl" \
  --run 0.07=20260980="${ROOT}/gate007_seed20260980/${METHOD}.jsonl" \
  --run 0.08=20260977="${ROOT}/gate008_seed20260977/${METHOD}.jsonl" \
  --run 0.08=20260979="${ROOT}/gate008_seed20260979/${METHOD}.jsonl" \
  --run 0.08=20260980="${ROOT}/gate008_seed20260980/${METHOD}.jsonl" \
  --run 0.09=20260977="${ROOT}/gate009_seed20260977/${METHOD}.jsonl" \
  --run 0.09=20260979="${ROOT}/gate009_seed20260979/${METHOD}.jsonl" \
  --run 0.09=20260980="${ROOT}/gate009_seed20260980/${METHOD}.jsonl" \
  --run 0.10=20260977="${ROOT}/gate010_seed20260977/${METHOD}.jsonl" \
  --run 0.10=20260979="${ROOT}/gate010_seed20260979/${METHOD}.jsonl" \
  --run 0.10=20260980="${ROOT}/gate010_seed20260980/${METHOD}.jsonl" \
  --run 0.11=20260977="${ROOT}/gate011_seed20260977/${METHOD}.jsonl" \
  --run 0.11=20260979="${ROOT}/gate011_seed20260979/${METHOD}.jsonl" \
  --run 0.11=20260980="${ROOT}/gate011_seed20260980/${METHOD}.jsonl" \
  --run 0.13=20260977="${ROOT}/gate013_seed20260977/${METHOD}.jsonl" \
  --run 0.13=20260979="${ROOT}/gate013_seed20260979/${METHOD}.jsonl" \
  --run 0.13=20260980="${ROOT}/gate013_seed20260980/${METHOD}.jsonl" \
  --run 0.15=20260977="${ROOT}/gate015_seed20260977/${METHOD}.jsonl" \
  --run 0.15=20260979="${ROOT}/gate015_seed20260979/${METHOD}.jsonl" \
  --run 0.15=20260980="${ROOT}/gate015_seed20260980/${METHOD}.jsonl"

PYTHONPATH=src:. "${PY}" scripts/bootstrap_grpo_matrix.py \
  "${BASELINE}" "${REPORT}/bootstrap_gate009" \
  --bootstrap 5000 --seed 20260802 \
  --run lambda025_gate009=20260977="${ROOT}/gate009_seed20260977/${METHOD}.jsonl" \
  --run lambda025_gate009=20260979="${ROOT}/gate009_seed20260979/${METHOD}.jsonl" \
  --run lambda025_gate009=20260980="${ROOT}/gate009_seed20260980/${METHOD}.jsonl"

date -Is >"${REPORT}/PIPELINE_COMPLETED"
