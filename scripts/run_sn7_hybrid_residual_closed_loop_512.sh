#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
DATA="${STORE}/processed/sn7_v1/agent/sequential_selector_v1/full"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
STATES="${RUN}/current_policy_val_collection_512_20260801/states_val_512.jsonl"
OUT="${RUN}/hybrid_residual_closed_loop_512_seed20260813"

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${OUT}"

run_shard() {
  local gpu="$1" shard="$2"
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${ADAPTER}" "${STATES}" \
    "${DATA}/episodes_train_val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${OUT}/shard${shard}" \
    --gpu "${gpu}" --seed 20260813 --split val \
    --policy-mode hybrid_residual_ranker \
    --ranker-checkpoint "${RUN}/online_hybrid_residual_2k_seed20260813/best.pt" \
    --utility-head "${RUN}/vla_utility_head_shared_last_full/gate.joblib" \
    --utility-head-summary "${RUN}/vla_utility_head_shared_last_full/summary.json" \
    --tool-mode selective --belief-mode recurrent \
    --tool-belief-checkpoint "${STORE}/runs/agent/sn7_step0_tool_belief_gated_seed20260730/best_promoted.pt" \
    --tool-gate "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/gate.joblib" \
    --tool-gate-summary "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/summary.json" \
    --tool-artifact-root "${OUT}/shard${shard}_tools" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --max-candidates 16 --max-acquisitions 2 \
    --bootstrap-repetitions 0 --limit 256 \
    --num-shards 2 --shard-index "${shard}" --monitor-interval 5
}

run_shard 1 0 >"${OUT}/shard0.log" 2>&1 & p0=$!
run_shard 4 1 >"${OUT}/shard1.log" 2>&1 & p1=$!
wait "${p0}"
wait "${p1}"
