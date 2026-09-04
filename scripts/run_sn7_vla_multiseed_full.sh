#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
RUN="${STORAGE_ROOT}/runs/sn7_active_catalog"
DATA="${RUN}/active_catalog_rl_states"
CATALOG="${STORAGE_ROOT}/processed/sn7_v1/agent/sequential_selector_v1/full"
RANKER="${RUN}/candidate_ranker_v2/seed20260720/best.pt"
LOG_ROOT="${STORAGE_ROOT}/logs/sn7_vla_multiseed_full_20260728"
GPUS="${GPUS:-0,1,2,3}"
SEEDS=(20260717 20260718 20260719)

cd "${PROJECT_ROOT}"
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${LOG_ROOT}"

IFS=',' read -ra GPU_LIST <<< "${GPUS}"
[[ "${#GPU_LIST[@]}" -eq 4 ]] || {
  echo "exactly four GPU ids are required" >&2
  exit 2
}
for gpu in "${GPU_LIST[@]}"; do
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} is occupied: ${pids}" >&2
    exit 3
  }
done

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
    *)
      echo "unknown seed: $1" >&2
      return 1
      ;;
  esac
}

extract_shard() {
  local seed="$1"
  local adapter="$2"
  local split="$3"
  local shard="$4"
  local gpu="$5"
  local output="${RUN}/vla_features_seed${seed}_last_${split}_full_s${shard}of3"
  if [[ -s "${output}/summary.json" ]]; then
    return
  fi
  [[ ! -e "${output}" ]] || {
    echo "refusing partial feature shard: ${output}" >&2
    return 1
  }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" \
    scripts/extract_active_catalog_vla_features.py \
    "${MODEL}" "${adapter}" "${DATA}/${split}.jsonl" "${output}" \
    --device cuda:0 --batch-size 2 --pooling last \
    --num-shards 3 --shard-index "${shard}" \
    >"${LOG_ROOT}/seed${seed}_${split}_s${shard}.log" 2>&1
}

merge_split() {
  local seed="$1"
  local split="$2"
  local output="${RUN}/vla_features_seed${seed}_last_${split}_full"
  if [[ -s "${output}/summary.json" ]]; then
    return
  fi
  [[ ! -e "${output}" ]] || {
    echo "refusing partial merged features: ${output}" >&2
    return 1
  }
  "${PYTHON}" scripts/merge_active_catalog_vla_features.py "${output}" \
    "${RUN}/vla_features_seed${seed}_last_${split}_full_s0of3" \
    "${RUN}/vla_features_seed${seed}_last_${split}_full_s1of3" \
    "${RUN}/vla_features_seed${seed}_last_${split}_full_s2of3" \
    >"${LOG_ROOT}/seed${seed}_${split}_merge.log" 2>&1
}

train_head() {
  local seed="$1"
  local output="${RUN}/vla_utility_head_seed${seed}_last_full"
  if [[ -s "${output}/summary.json" ]]; then
    return
  fi
  [[ ! -e "${output}" ]] || {
    echo "refusing partial utility head: ${output}" >&2
    return 1
  }
  OMP_NUM_THREADS=8 "${PYTHON}" scripts/train_visual_utility_gate.py \
    "${RUN}/vla_features_seed${seed}_last_train_full" \
    "${RUN}/vla_features_seed${seed}_last_val_full" \
    "${output}" --seed "${seed}" --max-call-rate 0.15 --min-oof-recall 0.10 \
    >"${LOG_ROOT}/seed${seed}_utility_head.log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  adapter="$(adapter_for_seed "${seed}")"
  [[ -s "${adapter}/adapter_config.json" ]] || {
    echo "missing adapter for seed ${seed}: ${adapter}" >&2
    exit 4
  }
  extract_shard "${seed}" "${adapter}" train 0 "${GPU_LIST[0]}" &
  pid0="$!"
  extract_shard "${seed}" "${adapter}" train 1 "${GPU_LIST[1]}" &
  pid1="$!"
  extract_shard "${seed}" "${adapter}" train 2 "${GPU_LIST[2]}" &
  pid2="$!"
  (
    extract_shard "${seed}" "${adapter}" val 0 "${GPU_LIST[3]}"
    extract_shard "${seed}" "${adapter}" val 1 "${GPU_LIST[3]}"
    extract_shard "${seed}" "${adapter}" val 2 "${GPU_LIST[3]}"
  ) &
  pid3="$!"
  status=0
  for pid in "${pid0}" "${pid1}" "${pid2}" "${pid3}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  [[ "${status}" -eq 0 ]] || exit "${status}"
  merge_split "${seed}" train
  merge_split "${seed}" val
  train_head "${seed}"
done

run_closed_loop() {
  local seed="$1"
  local gpu="$2"
  local adapter head output
  adapter="$(adapter_for_seed "${seed}")"
  head="${RUN}/vla_utility_head_seed${seed}_last_full"
  output="${RUN}/closed_loop_vla_full_seed${seed}"
  [[ ! -e "${output}" ]] || {
    echo "refusing existing closed-loop output: ${output}" >&2
    return 1
  }
  "${PYTHON}" scripts/launch_active_catalog_closed_loop.py \
    "${MODEL}" "${adapter}" \
    "${CATALOG}/closed_loop_v1/states_val_step0.jsonl" \
    "${CATALOG}/closed_loop_v1/episodes_val.jsonl" \
    "${CATALOG}/active_catalog_sft_v4/val.jsonl" \
    "${CATALOG}/active_catalog_sft_v4/val_evaluation_index.jsonl" \
    "${output}" --gpu "${gpu}" --seed "${seed}" \
    --policy-mode utility_head_ranker \
    --ranker-checkpoint "${RANKER}" \
    --utility-head "${head}/gate.joblib" \
    --utility-head-summary "${head}/summary.json" \
    --max-candidates 16 --max-acquisitions 2 \
    --bootstrap-repetitions 2000 --monitor-interval 5 \
    >"${LOG_ROOT}/seed${seed}_closed_loop.log" 2>&1
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
    "${seed}=${RUN}/closed_loop_vla_full_seed${seed}/evaluation/traces.jsonl"
  )
done
for baseline in always_stop uncertainty_gate; do
  "${PYTHON}" scripts/aggregate_active_catalog_policy_seeds.py \
    "${RUN}/vla_full_three_seed_vs_${baseline}.json" \
    "${candidate_args[@]}" \
    --reference "${RUN}/closed_loop_baselines/${baseline}.jsonl" \
    --repetitions 5000 --seed 20260728 \
    >"${LOG_ROOT}/aggregate_vs_${baseline}.log" 2>&1
done
