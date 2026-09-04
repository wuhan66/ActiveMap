#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/home/wh/ActiveMap}"
MANIFEST="${STORAGE_ROOT}/processed/sn7_v1/updater_v4_cap20/updater_samples.jsonl"
CHANGE_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/change_mamba"
BAN_ROOT="${STORAGE_ROOT}/runs/external_baselines/sn7/open_cd_ban"
CHANGE_REPO="${STORAGE_ROOT}/external/change_mamba"
CHANGE_CONFIG="${CHANGE_REPO}/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
OPENCD_REPO="${STORAGE_ROOT}/external/open_cd"
OPENCD_CONFIG="${OPENCD_REPO}/configs/ban/ban_vit-b16-clip_mit-b0_512x512_40k_levircd.py"
CHANGE_PYTHON="${STORAGE_ROOT}/envs/activemap-change-mamba/bin/python"
BAN_PYTHON="${STORAGE_ROOT}/envs/activemap-opencd/bin/python"
RUN_ROOT="${RUN_ROOT:-${STORAGE_ROOT}/runs/frozen_test/sn7_safe_commit_20260727}"
LOG_ROOT="${RUN_ROOT}/logs"
GPUS="${GPUS:-0,1,4,5}"
SEEDS=(20260725 20260726 20260727)

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
"${BAN_PYTHON}" -c \
  'from activemap.frozen_test import assert_frozen_test_access; assert_frozen_test_access()'
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "refusing to reuse frozen test root: ${RUN_ROOT}" >&2
  exit 41
}
IFS=',' read -ra GPU_LIST <<< "${GPUS}"
[[ "${#GPU_LIST[@]}" -eq 4 ]] || {
  echo "exactly four GPU ids are required" >&2
  exit 42
}
for gpu in "${GPU_LIST[@]}"; do
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} is occupied: ${pids}" >&2
    exit 43
  }
done
mkdir -p "${LOG_ROOT}"

run_change_test() {
  local seed="$1"
  local gpu="$2"
  local run="${CHANGE_ROOT}/full_weight5_seed${seed}_v1"
  CUDA_VISIBLE_DEVICES="${gpu}" "${CHANGE_PYTHON}" \
    scripts/evaluate_sn7_changemamba.py \
    "${run}/best.pt" "${MANIFEST}" "${CHANGE_REPO}" "${CHANGE_CONFIG}" \
    "${RUN_ROOT}/predictions/changemamba_seed${seed}" \
    --split test --device cuda:0 --batch-size 16 --workers 8 \
    >"${LOG_ROOT}/changemamba_seed${seed}.log" 2>&1
}

run_ban_test() {
  local seed="$1"
  local gpu="$2"
  local run="${BAN_ROOT}/full_weight5_seed${seed}_v1"
  CUDA_VISIBLE_DEVICES="${gpu}" "${BAN_PYTHON}" \
    scripts/evaluate_sn7_opencd_ban.py \
    "${run}/best.pt" "${MANIFEST}" "${OPENCD_REPO}" "${OPENCD_CONFIG}" \
    "${RUN_ROOT}/predictions/ban_seed${seed}" \
    --split test --device cuda:0 --batch-size 8 --workers 8 \
    >"${LOG_ROOT}/ban_seed${seed}.log" 2>&1
}

pids=()
for index in 0 1 2; do
  run_change_test "${SEEDS[${index}]}" "${GPU_LIST[${index}]}" &
  pids+=("$!")
done
run_ban_test "${SEEDS[0]}" "${GPU_LIST[3]}" &
pids+=("$!")
for pid in "${pids[@]}"; do
  wait "${pid}"
done

pids=()
run_ban_test "${SEEDS[1]}" "${GPU_LIST[0]}" &
pids+=("$!")
run_ban_test "${SEEDS[2]}" "${GPU_LIST[1]}" &
pids+=("$!")
for pid in "${pids[@]}"; do
  wait "${pid}"
done

evaluate_policy() {
  local source="$1"
  local target="$2"
  local mode="$3"
  local seed="$4"
  local source_root target_root
  if [[ "${source}" == "changemamba" ]]; then
    source_root="${CHANGE_ROOT}"
  else
    source_root="${BAN_ROOT}"
  fi
  if [[ "${target}" == "changemamba" ]]; then
    target_root="${CHANGE_ROOT}"
  else
    target_root="${BAN_ROOT}"
  fi
  "${BAN_PYTHON}" scripts/evaluate_sn7_frozen_safe_commit.py \
    "${source_root}/full_weight5_seed${seed}_v1/train_audit/per_sample.jsonl" \
    "${target_root}/full_weight5_seed${seed}_v1/train_audit/per_sample.jsonl" \
    "${RUN_ROOT}/predictions/${target}_seed${seed}/per_sample.jsonl" \
    "${RUN_ROOT}/policies/${source}_to_${target}_${mode}_seed${seed}" \
    --source-backend "${source}" --target-backend "${target}" \
    --threshold-mode "${mode}" --folds 5 --l2 0.01 \
    >"${LOG_ROOT}/${source}_to_${target}_${mode}_seed${seed}.log" 2>&1
}

for seed in "${SEEDS[@]}"; do
  evaluate_policy changemamba changemamba source_oof "${seed}"
  evaluate_policy ban ban source_oof "${seed}"
  evaluate_policy changemamba ban source_oof "${seed}"
  evaluate_policy changemamba ban all_target_train "${seed}"
  evaluate_policy ban changemamba source_oof "${seed}"
  evaluate_policy ban changemamba all_target_train "${seed}"
done

for policy in \
  changemamba_to_changemamba_source_oof \
  ban_to_ban_source_oof \
  changemamba_to_ban_source_oof \
  changemamba_to_ban_all_target_train \
  ban_to_changemamba_source_oof \
  ban_to_changemamba_all_target_train
do
  runs=()
  for seed in "${SEEDS[@]}"; do
    runs+=("${RUN_ROOT}/policies/${policy}_seed${seed}")
  done
  "${BAN_PYTHON}" scripts/bootstrap_sn7_changemamba_safe_commit.py \
    "${RUN_ROOT}/bootstrap/${policy}.json" "${runs[@]}" \
    --draws 5000 --seed 20260727 --frozen-test \
    --output-markdown "${RUN_ROOT}/bootstrap/${policy}.md" \
    >"${LOG_ROOT}/bootstrap_${policy}.log" 2>&1
done
