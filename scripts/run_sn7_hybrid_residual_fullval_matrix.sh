#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
DATA="${STORE}/processed/sn7_v1/agent/sequential_selector_v1/full"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
STATES="${DATA}/closed_loop_v1/states_val_step0.jsonl"
ROOT="${RUN}/hybrid_residual_fullval_matrix_20260801"
GPUS=(1 2 4 7)

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${ROOT}"

completed() {
  local result="$1/process_result.json"
  [[ -f "${result}" ]] && grep -q '"status": "completed"' "${result}"
}

run_shard() {
  local label="$1" mode="$2" ranker="$3" seed="$4" shard="$5" gpu="$6"
  local out="${ROOT}/${label}/shard${shard}"
  if completed "${out}"; then
    echo "skip completed ${label} shard${shard}"
    return 0
  fi
  if [[ -e "${out}" ]]; then
    echo "refusing incomplete existing shard: ${out}" >&2
    return 20
  fi
  args=(
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py
    "${MODEL}" "${ADAPTER}" "${STATES}"
    "${DATA}/episodes_train_val.jsonl"
    "${DATA}/active_catalog_sft_v4/val.jsonl"
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl"
    "${out}"
    --gpu "${gpu}" --seed "${seed}" --split val
    --policy-mode "${mode}" --ranker-checkpoint "${ranker}"
    --utility-head "${RUN}/vla_utility_head_shared_last_full/gate.joblib"
    --utility-head-summary "${RUN}/vla_utility_head_shared_last_full/summary.json"
    --tool-mode selective --belief-mode recurrent
    --tool-belief-checkpoint "${STORE}/runs/agent/sn7_step0_tool_belief_gated_seed20260730/best_promoted.pt"
    --tool-gate "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/gate.joblib"
    --tool-gate-summary "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/summary.json"
    --tool-artifact-root "${ROOT}/${label}/shard${shard}_tools"
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}"
    --max-candidates 16 --max-acquisitions 2
    --bootstrap-repetitions 0 --num-shards 4 --shard-index "${shard}"
    --monitor-interval 5
  )
  "${args[@]}"
}

run_variant() {
  local label="$1" mode="$2" ranker="$3" seed="$4"
  mkdir -p "${ROOT}/${label}"
  pids=()
  for shard in 0 1 2 3; do
    run_shard "${label}" "${mode}" "${ranker}" "${seed}" "${shard}" "${GPUS[$shard]}" \
      >"${ROOT}/${label}/shard${shard}.log" 2>&1 &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then status=1; fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "variant failed: ${label}" >&2
    return "${status}"
  fi
  mkdir -p "${ROOT}/${label}/merged"
  cat "${ROOT}/${label}"/shard*/evaluation/traces.jsonl \
    >"${ROOT}/${label}/merged/traces.jsonl"
  local count
  count="$(wc -l <"${ROOT}/${label}/merged/traces.jsonl")"
  [[ "${count}" -eq 6369 ]] || {
    echo "unexpected merged trace count for ${label}: ${count}" >&2
    return 21
  }
}

run_variant \
  old_vla utility_head_ranker \
  "${RUN}/candidate_ranker_v2/seed20260720/best.pt" 20260801

for seed in 20260811 20260812 20260813; do
  run_variant \
    "hybrid_seed${seed}" hybrid_residual_ranker \
    "${RUN}/online_hybrid_residual_2k_seed${seed}/best.pt" "${seed}"
  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${ROOT}/hybrid_seed${seed}/paired_vs_old_vla.json" \
    --records "old_vla=${ROOT}/old_vla/merged/traces.jsonl" \
    --records "hybrid=${ROOT}/hybrid_seed${seed}/merged/traces.jsonl" \
    --candidate hybrid --repetitions 5000 --seed "${seed}"
done

"${PYTHON}" scripts/aggregate_active_catalog_paired_policy_seeds.py \
  "${ROOT}/hybrid_three_seed_vs_old_vla.json" \
  --pair "20260811=${ROOT}/hybrid_seed20260811/merged/traces.jsonl,${ROOT}/old_vla/merged/traces.jsonl" \
  --pair "20260812=${ROOT}/hybrid_seed20260812/merged/traces.jsonl,${ROOT}/old_vla/merged/traces.jsonl" \
  --pair "20260813=${ROOT}/hybrid_seed20260813/merged/traces.jsonl,${ROOT}/old_vla/merged/traces.jsonl" \
  --repetitions 10000 --seed 20260801

printf 'complete\n' >"${ROOT}/MATRIX_COMPLETE"
