#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_vla_multiseed_full_20260728"
POLL_SECONDS="${POLL_SECONDS:-120}"
SEEDS=(20260717 20260718 20260719)

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

candidate_args=()
for seed in "${SEEDS[@]}"; do
  trace="${RUN}/closed_loop_vla_full_seed${seed}/evaluation/traces.jsonl"
  until [[ -s "${trace}" ]]; do
    sleep "${POLL_SECONDS}"
  done
  candidate_args+=(--candidate "${seed}=${trace}")
done

for baseline in always_stop uncertainty_gate; do
  output="${RUN}/vla_full_three_seed_vs_${baseline}.json"
  [[ ! -e "${output}" ]] || continue
  "${PYTHON}" scripts/aggregate_active_catalog_policy_seeds.py \
    "${output}" "${candidate_args[@]}" \
    --reference "${RUN}/closed_loop_baselines/${baseline}.jsonl" \
    --repetitions 5000 --seed 20260728 \
    >"${LOG_ROOT}/aggregate_vs_${baseline}.log" 2>&1
done
