#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
DATA="${STORE}/processed/sn7_v1/agent/sequential_selector_v1/full"
SOURCE="${STORE}/processed/sn7_v1/agent/sequential_selector_v1/selector_states_train_val.jsonl"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
ROOT="${RUN}/current_policy_train_collection_8k_20260801"
STATES="${ROOT}/states_train_8192.jsonl"
GPUS=(1 2 4 7)

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${ROOT}"

if [[ ! -f "${STATES}" ]]; then
  "${PYTHON}" scripts/sample_active_catalog_online_states.py \
    "${SOURCE}" "${STATES}" --split train --count 8192 --seed 20260821
fi

completed() {
  local result="$1/process_result.json"
  [[ -f "${result}" ]] && grep -q '"status": "completed"' "${result}"
}

run_collection_shard() {
  local shard="$1"
  local gpu="$2"
  local out="${ROOT}/shard${shard}"
  if completed "${out}"; then
    echo "skip completed collection shard${shard}"
    return 0
  fi
  if [[ -e "${out}" ]]; then
    echo "refusing incomplete existing collection shard: ${out}" >&2
    return 20
  fi
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${ADAPTER}" "${STATES}" \
    "${DATA}/episodes_train_val.jsonl" \
    "${DATA}/active_catalog_sft_v4/train.jsonl" \
    "${DATA}/active_catalog_sft_v4/train_evaluation_index.jsonl" \
    "${out}" \
    --gpu "${gpu}" --seed 20260821 --split train \
    --policy-mode utility_head_ranker \
    --ranker-checkpoint "${RUN}/candidate_ranker_v2/seed20260720/best.pt" \
    --utility-head "${RUN}/vla_utility_head_shared_last_full/gate.joblib" \
    --utility-head-summary "${RUN}/vla_utility_head_shared_last_full/summary.json" \
    --tool-mode selective --belief-mode recurrent \
    --tool-belief-checkpoint "${STORE}/runs/agent/sn7_step0_tool_belief_gated_seed20260730/best_promoted.pt" \
    --tool-gate "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/gate.joblib" \
    --tool-gate-summary "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/summary.json" \
    --tool-artifact-root "${ROOT}/shard${shard}_tools" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --max-candidates 16 --max-acquisitions 2 --bootstrap-repetitions 0 \
    --num-shards 4 --shard-index "${shard}" --monitor-interval 5
}

pids=()
for shard in 0 1 2 3; do
  run_collection_shard "${shard}" "${GPUS[$shard]}" \
    >"${ROOT}/shard${shard}.log" 2>&1 &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" -eq 0 ]] || exit "${status}"

mkdir -p "${ROOT}/merged"
cat "${ROOT}"/shard*/evaluation/traces.jsonl >"${ROOT}/merged/traces.jsonl"
count="$(wc -l <"${ROOT}/merged/traces.jsonl")"
[[ "${count}" -eq 8192 ]] || {
  echo "unexpected merged train count: ${count}" >&2
  exit 21
}

VAL_TRACES="${RUN}/hybrid_residual_fullval_matrix_20260801/old_vla/merged/traces.jsonl"
TRAIN_INDEX="${DATA}/active_catalog_sft_v4/train_evaluation_index.jsonl"
VAL_INDEX="${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl"

train_seed() {
  local seed="$1"
  local gpu="$2"
  local out="${RUN}/online_hybrid_residual_8k_seed${seed}"
  if [[ -f "${out}/summary.json" ]]; then
    echo "skip completed training seed${seed}"
    return 0
  fi
  if [[ -e "${out}" ]]; then
    echo "refusing incomplete existing training output: ${out}" >&2
    return 22
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/train_active_catalog_candidate_ranker.py \
    "${ROOT}/merged/traces.jsonl" "${TRAIN_INDEX}" \
    "${VAL_TRACES}" "${VAL_INDEX}" "${out}" \
    --input-format online_trace \
    --online-event-feature predicted_acquire_utility \
    --device cuda:0 --seed "${seed}" --epochs 60 --patience 10 \
    --pairwise-weight 1 --maximum-false-call-rate 0.05
}

train_seed 20260821 1 >"${ROOT}/train_seed20260821.log" 2>&1 & p1=$!
train_seed 20260822 2 >"${ROOT}/train_seed20260822.log" 2>&1 & p2=$!
train_seed 20260823 4 >"${ROOT}/train_seed20260823.log" 2>&1 & p3=$!

status=0
for pid in "${p1}" "${p2}" "${p3}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" -eq 0 ]] || exit "${status}"
printf 'complete\n' >"${ROOT}/TRAINING_COMPLETE"
