#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
WITH_TOOLS="${RUN}/hybrid_residual_8k_fullval_matrix_20260801"
WITHOUT_TOOLS="${RUN}/hybrid_residual_8k_no_tools_fullval_matrix_20260801"
OUTPUT="${WITHOUT_TOOLS}/with_tools_minus_no_tools_three_seed.json"
POLL_SECONDS="${POLL_SECONDS:-120}"

for seed in 20260821 20260822 20260823; do
  result="${WITHOUT_TOOLS}/seed${seed}/process_result.json"
  until [[ -s "${result}" ]] && grep -q '"status": "completed"' "${result}"; do
    sleep "${POLL_SECONDS}"
  done
done

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
if [[ ! -e "${OUTPUT}" ]]; then
  "${PYTHON}" scripts/aggregate_active_catalog_paired_policy_seeds.py \
    "${OUTPUT}" \
    --pair "20260821=${WITH_TOOLS}/seed20260821/evaluation/traces.jsonl,${WITHOUT_TOOLS}/seed20260821/evaluation/traces.jsonl" \
    --pair "20260822=${WITH_TOOLS}/seed20260822/evaluation/traces.jsonl,${WITHOUT_TOOLS}/seed20260822/evaluation/traces.jsonl" \
    --pair "20260823=${WITH_TOOLS}/seed20260823/evaluation/traces.jsonl,${WITHOUT_TOOLS}/seed20260823/evaluation/traces.jsonl" \
    --repetitions 10000 --seed 20260801
fi
printf 'complete\n' >"${WITHOUT_TOOLS}/MATRIX_COMPLETE"
