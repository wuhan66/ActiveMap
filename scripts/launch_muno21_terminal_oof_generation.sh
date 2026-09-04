#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${ACTIVEMAP_STORAGE_ROOT:-/home/wh/ActiveMap}"
PYTHON="${STORAGE_ROOT}/envs/activemap-agent/bin/python"
MODEL="/home/wh/hf_models/Qwen3-VL-4B-Instruct"
FOLDS="${STORAGE_ROOT}/processed/muno21_v2/agent/policy_relative_crossfit_qwen_v1"
TRAINING="${STORAGE_ROOT}/runs/agent_sft/qwen3_vl_4b_policy_relative_crossfit_v1"
OUTPUT="${STORAGE_ROOT}/runs/agent_eval/qwen3_vl_4b_terminal_oof_predictions_v1"
LOGS="${STORAGE_ROOT}/logs/qwen3_vl_4b_terminal_oof_predictions_v1"

mkdir -p "${OUTPUT}" "${LOGS}"
cd "${PROJECT_ROOT}"
gpus=(1 2 3)
for fold in 0 1 2; do
  gpu="${gpus[$fold]}"
  adapter="${TRAINING}/fold${fold}/seed20260716/final"
  data="${FOLDS}/fold${fold}/holdout_rollout.jsonl"
  output="${OUTPUT}/fold${fold}"
  log="${LOGS}/fold${fold}.log"
  [[ -s "${adapter}/adapter_model.safetensors" ]] || {
    echo "missing adapter: ${adapter}" >&2
    exit 3
  }
  [[ ! -e "${output}" ]] || {
    echo "refusing to overwrite ${output}" >&2
    exit 3
  }
  gpu_pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${gpu_pids//[[:space:]]/}" ]] || {
    echo "GPU${gpu} occupied: ${gpu_pids}" >&2
    exit 4
  }
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}" \
    "${PYTHON}" scripts/evaluate_semantic_vlm_actions.py \
    "${MODEL}" "${adapter}" "${data}" "${output}" --seed 20260716 \
    > "${log}" 2>&1 < /dev/null &
  echo "$!" > "${LOGS}/fold${fold}.pid"
  echo "started fold=${fold} gpu=${gpu} pid=$!"
done
