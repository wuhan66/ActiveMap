#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${STORAGE_ROOT}/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed3/seed20260719/final"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"

run_variant() {
  local gpu="$1"
  local output="$2"
  local ranker="$3"
  local head="$4"
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${ADAPTER}" \
    "${DATA}/closed_loop_v1/states_val_step0.jsonl" \
    "${DATA}/closed_loop_v1/episodes_val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${RUN}/${output}" \
    --gpu "${gpu}" --seed 20260727 \
    --policy-mode utility_head_ranker \
    --ranker-checkpoint "${RUN}/${ranker}/seed20260720/best.pt" \
    --utility-head "${RUN}/${head}/gate.joblib" \
    --utility-head-summary "${RUN}/${head}/summary.json" \
    --max-candidates 16 --max-acquisitions 2 \
    --bootstrap-repetitions 2000 --monitor-interval 5
}

run_variant 1 closed_loop_vla_head_last_ranker_v4_full_val \
  candidate_ranker_v4 vla_utility_head_last_pilot &
pid1="$!"
run_variant 4 closed_loop_vla_head_last_ranker_v2_full_val \
  candidate_ranker_v2 vla_utility_head_last_pilot &
pid2="$!"
run_variant 6 closed_loop_vla_head_last_mean_ranker_v4_full_val \
  candidate_ranker_v4 vla_utility_head_last_mean_pilot &
pid3="$!"

status=0
for pid in "${pid1}" "${pid2}" "${pid3}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
exit "${status}"
