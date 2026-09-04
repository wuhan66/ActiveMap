#!/usr/bin/env bash
set -euo pipefail

PROJECT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORE="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORE}/envs/activemap-agent/bin/python"
RUN="${STORE}/runs/sn7_active_catalog"
DATA="${STORE}/processed/sn7_v1/agent/sequential_selector_v1/full"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
ADAPTER="${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
ROOT="${RUN}/remaining_protocol_fullval_matrix_20260801"
POLL_SECONDS="${POLL_SECONDS:-60}"

while [[ ! -s "${RUN}/hybrid_residual_8k_fullval_matrix_20260801/MATRIX_COMPLETE" ]]; do
  sleep "${POLL_SECONDS}"
done

for gpu in 1 2 4; do
  while [[ -n "$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)" ]]; do
    sleep "${POLL_SECONDS}"
  done
done

cd "${PROJECT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
mkdir -p "${ROOT}"

completed() {
  local result="$1/process_result.json"
  [[ -f "${result}" ]] && grep -q '"status": "completed"' "${result}"
}

run_protocol() {
  local label="$1"
  local mode="$2"
  local tool_mode="$3"
  local gpu="$4"
  local out="${ROOT}/${label}"
  if completed "${out}"; then
    echo "skip completed ${label}"
    return 0
  fi
  if [[ -e "${out}" ]]; then
    echo "refusing incomplete output: ${out}" >&2
    return 20
  fi
  args=(
    "${PYTHON}" scripts/launch_active_catalog_closed_loop.py
    "${MODEL}" "${ADAPTER}"
    "${DATA}/closed_loop_v1/states_val_step0.jsonl"
    "${DATA}/episodes_train_val.jsonl"
    "${DATA}/active_catalog_sft_v4/val.jsonl"
    "${DATA}/active_catalog_sft_v4/val_evaluation_index.jsonl"
    "${out}"
    --gpu "${gpu}" --seed 20260718 --split val
    --policy-mode "${mode}" --tool-mode "${tool_mode}" --belief-mode recurrent
    --max-candidates 16 --max-acquisitions 2 --max-new-tokens 96
    --bootstrap-repetitions 5000 --monitor-interval 5
  )
  if [[ "${tool_mode}" == "model" ]]; then
    args+=(
      --tool-belief-checkpoint
      "${STORE}/runs/agent/sn7_step0_tool_belief_gated_seed20260730/best_promoted.pt"
      --tool-artifact-root "${ROOT}/${label}_tools"
      --asset-root-map "/mnt/mydisk/wh/ActiveMap=${STORE}"
    )
  fi
  "${args[@]}"
}

run_protocol geommagent geommagent_style model 1 >"${ROOT}/geommagent.log" 2>&1 & p1=$!
run_protocol sensesearch sensesearch_style model 2 >"${ROOT}/sensesearch.log" 2>&1 & p2=$!
run_protocol plan_execute plan_execute none 4 >"${ROOT}/plan_execute.log" 2>&1 & p3=$!

status=0
for pid in "${p1}" "${p2}" "${p3}"; do
  if ! wait "${pid}"; then status=1; fi
done
[[ "${status}" -eq 0 ]] || exit "${status}"
printf 'complete\n' >"${ROOT}/PROTOCOLS_COMPLETE"
