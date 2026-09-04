#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
CATALOG="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
RANKER="${RUN}/candidate_ranker_v2/seed20260720/best.pt"
HEAD="${RUN}/vla_utility_head_shared_last_full"
THRESHOLD="${THRESHOLD:-0.1291167289018631}"
TAG="${TAG:-risk_t0129}"
GPUS="${GPUS:-0,1,2}"
POLL_SECONDS="${POLL_SECONDS:-120}"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_vla_shared_head_${TAG}_20260728"
SEEDS=(20260717 20260718 20260719)

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${LOG_ROOT}"
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
[[ "${#GPU_LIST[@]}" -eq 3 ]] || {
  echo "exactly three GPU ids are required" >&2
  exit 2
}

adapter_for_seed() {
  case "$1" in
    20260717)
      echo "${RUN}/qwen3vl4b_seed1_eval500/seed20260717/final"
      ;;
    20260718)
      echo "${RUN}/qwen3vl4b_weighted_replication_seed2/seed20260718/final"
      ;;
    20260719)
      echo "${RUN}/qwen3vl4b_weighted_replication_seed3/seed20260719/final"
      ;;
  esac
}

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local pids
    pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
    [[ -z "${pids//[[:space:]]/}" ]] && return
    sleep "${POLL_SECONDS}"
  done
}

run_closed_loop() {
  local seed="$1"
  local gpu="$2"
  local adapter output
  adapter="$(adapter_for_seed "${seed}")"
  output="${RUN}/closed_loop_vla_shared_head_${TAG}_seed${seed}"
  [[ ! -e "${output}" ]] || {
    echo "refusing existing threshold output: ${output}" >&2
    return 1
  }
  wait_for_gpu "${gpu}"
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${adapter}" \
    "${CATALOG}/closed_loop_v1/states_val_step0.jsonl" \
    "${CATALOG}/closed_loop_v1/episodes_val.jsonl" \
    "${CATALOG}/active_catalog_sft_v4/val.jsonl" \
    "${CATALOG}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${output}" --gpu "${gpu}" --seed "${seed}" \
    --policy-mode utility_head_ranker \
    --ranker-checkpoint "${RANKER}" \
    --utility-head "${HEAD}/gate.joblib" \
    --utility-head-summary "${HEAD}/summary.json" \
    --utility-threshold-override "${THRESHOLD}" \
    --max-candidates 16 --max-acquisitions 2 \
    --bootstrap-repetitions 2000 --monitor-interval 5 \
    >"${LOG_ROOT}/closed_loop_seed${seed}.log" 2>&1
}

pids=()
for index in 0 1 2; do
  run_closed_loop "${SEEDS[${index}]}" "${GPU_LIST[${index}]}" &
  pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
[[ "${status}" -eq 0 ]] || exit "${status}"

candidate_args=()
for seed in "${SEEDS[@]}"; do
  candidate_args+=(
    --candidate
    "${seed}=${RUN}/closed_loop_vla_shared_head_${TAG}_seed${seed}/evaluation/traces.jsonl"
  )
done
for baseline in always_stop uncertainty_gate; do
  "${PYTHON}" scripts/aggregate_active_catalog_policy_seeds.py \
    "${RUN}/vla_shared_head_${TAG}_three_seed_vs_${baseline}.json" \
    "${candidate_args[@]}" \
    --reference "${RUN}/closed_loop_baselines/${baseline}.jsonl" \
    --repetitions 5000 --seed 20260728 \
    >"${LOG_ROOT}/aggregate_vs_${baseline}.log" 2>&1
done
