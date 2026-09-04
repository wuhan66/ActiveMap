#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${STORAGE_ROOT}/runs/sn7_active_catalog/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
SOURCE_STATES="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/selector_states_train_val.jsonl"
OUT_ROOT="${RUN}/current_policy_val_collection_512_20260801"
STATES="${OUT_ROOT}/states_val_512.jsonl"

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}"
if [[ ! -f "${STATES}" ]]; then
  "${PYTHON}" scripts/sample_active_catalog_online_states.py \
    "${SOURCE_STATES}" "${STATES}" \
    --split val --count 512 --seed 20260801
fi

run_shard() {
  local gpu="$1" shard="$2"
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${ADAPTER}" "${STATES}" \
    "${DATA}/episodes_train_val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${OUT_ROOT}/shard${shard}" \
    --gpu "${gpu}" --seed 20260801 --split val \
    --policy-mode utility_head_ranker \
    --ranker-checkpoint "${RUN}/candidate_ranker_v2/seed20260720/best.pt" \
    --utility-head "${RUN}/vla_utility_head_shared_last_full/gate.joblib" \
    --utility-head-summary "${RUN}/vla_utility_head_shared_last_full/summary.json" \
    --tool-mode selective --belief-mode recurrent \
    --tool-belief-checkpoint "${STORAGE_ROOT}/runs/agent/sn7_step0_tool_belief_gated_seed20260730/best_promoted.pt" \
    --tool-gate "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/gate.joblib" \
    --tool-gate-summary "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/summary.json" \
    --tool-artifact-root "${OUT_ROOT}/shard${shard}_tools" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORAGE_ROOT}" \
    --max-candidates 16 --max-acquisitions 2 \
    --bootstrap-repetitions 0 --limit 256 \
    --num-shards 2 --shard-index "${shard}" --monitor-interval 5
}

run_shard 1 0 >"${OUT_ROOT}/shard0.log" 2>&1 & p0=$!
run_shard 4 1 >"${OUT_ROOT}/shard1.log" 2>&1 & p1=$!
wait "${p0}"
wait "${p1}"
