#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
RECURRENT="${RUN}/hybrid_residual_8k_fullval_matrix_20260801"
IDENTITY="${RUN}/hybrid_residual_8k_tool_belief_ablation_20260801/identity"
OUTPUT="${IDENTITY}/recurrent_minus_identity_three_seed.json"
POLL_SECONDS="${POLL_SECONDS:-120}"

for seed in 20260821 20260822 20260823; do
  result="${IDENTITY}/seed${seed}/process_result.json"
  until [[ -s "${result}" ]] && grep -q '"status": "completed"' "${result}"; do
    sleep "${POLL_SECONDS}"
  done
done

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
if [[ ! -e "${OUTPUT}" ]]; then
  "${PYTHON}" scripts/aggregate_active_catalog_paired_policy_seeds.py \
    "${OUTPUT}" \
    --pair "20260821=${RECURRENT}/seed20260821/evaluation/traces.jsonl,${IDENTITY}/seed20260821/evaluation/traces.jsonl" \
    --pair "20260822=${RECURRENT}/seed20260822/evaluation/traces.jsonl,${IDENTITY}/seed20260822/evaluation/traces.jsonl" \
    --pair "20260823=${RECURRENT}/seed20260823/evaluation/traces.jsonl,${IDENTITY}/seed20260823/evaluation/traces.jsonl" \
    --repetitions 10000 --seed 20260801
fi
printf 'complete\n' >"${IDENTITY}/MATRIX_COMPLETE"
