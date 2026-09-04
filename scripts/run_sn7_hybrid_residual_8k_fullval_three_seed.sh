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
REFERENCE="${RUN}/hybrid_residual_fullval_matrix_20260801/old_vla/merged/traces.jsonl"
ROOT="${RUN}/hybrid_residual_8k_fullval_matrix_20260801"

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${ROOT}"

completed() {
  local result="$1/process_result.json"
  [[ -f "${result}" ]] && grep -q '"status": "completed"' "${result}"
}

run_seed() {
  local seed="$1"
  local gpu="$2"
  local checkpoint="${RUN}/online_hybrid_residual_8k_seed${seed}/best.pt"
  local out="${ROOT}/seed${seed}"
  if completed "${out}"; then
    echo "skip completed seed${seed}"
    return 0
  fi
  if [[ -e "${out}" ]]; then
    echo "refusing incomplete output: ${out}" >&2
    return 20
  fi
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${ADAPTER}" "${STATES}" \
    "${DATA}/episodes_train_val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val.jsonl" \
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${out}" \
    --gpu "${gpu}" --seed "${seed}" --split val \
    --policy-mode hybrid_residual_ranker --ranker-checkpoint "${checkpoint}" \
    --utility-head "${RUN}/vla_utility_head_shared_last_full/gate.joblib" \
    --utility-head-summary "${RUN}/vla_utility_head_shared_last_full/summary.json" \
    --tool-mode selective --belief-mode recurrent \
    --tool-belief-checkpoint "${STORE}/runs/agent/sn7_step0_tool_belief_gated_seed20260730/best_promoted.pt" \
    --tool-gate "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/gate.joblib" \
    --tool-gate-summary "${RUN}/step0_tool_need_gate_seed20260730_benefit_gate_v1/summary.json" \
    --tool-artifact-root "${ROOT}/seed${seed}_tools" \
    --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}" \
    --max-candidates 16 --max-acquisitions 2 \
    --bootstrap-repetitions 0 --monitor-interval 5
}

run_seed 20260821 1 >"${ROOT}/seed20260821.log" 2>&1 & p1=$!
run_seed 20260822 2 >"${ROOT}/seed20260822.log" 2>&1 & p2=$!
run_seed 20260823 4 >"${ROOT}/seed20260823.log" 2>&1 & p3=$!

status=0
for pid in "${p1}" "${p2}" "${p3}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" -eq 0 ]] || exit "${status}"

for seed in 20260821 20260822 20260823; do
  traces="${ROOT}/seed${seed}/evaluation/traces.jsonl"
  count="$(wc -l <"${traces}")"
  [[ "${count}" -eq 6369 ]] || {
    echo "unexpected full-validation count for seed${seed}: ${count}" >&2
    exit 21
  }
  "${PYTHON}" scripts/compare_active_catalog_closed_loop.py \
    "${ROOT}/seed${seed}/paired_vs_old_vla.json" \
    --records "old_vla=${REFERENCE}" \
    --records "hybrid=${traces}" \
    --candidate hybrid --repetitions 5000 --seed "${seed}"
done

"${PYTHON}" scripts/aggregate_active_catalog_paired_policy_seeds.py \
  "${ROOT}/hybrid_three_seed_vs_old_vla.json" \
  --pair "20260821=${ROOT}/seed20260821/evaluation/traces.jsonl,${REFERENCE}" \
  --pair "20260822=${ROOT}/seed20260822/evaluation/traces.jsonl,${REFERENCE}" \
  --pair "20260823=${ROOT}/seed20260823/evaluation/traces.jsonl,${REFERENCE}" \
  --repetitions 10000 --seed 20260801

printf 'complete\n' >"${ROOT}/MATRIX_COMPLETE"
