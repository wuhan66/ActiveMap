#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
OVERLAY="${ACTIVEMAP_AGENT_SITE_PACKAGES:-${STORAGE_ROOT}/envs/agent_peft_overlay}"
MODEL="${MUNO21_DPO_MODEL:-${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821/checkpoints/checkpoint-200}"
DATA_ROOT="${MUNO21_AGENT_DATA_ROOT:-${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v6_anonymized}"
TRAIN_FILE="${DATA_ROOT}/train/safety_preferences.jsonl"
EVAL_FILE="${DATA_ROOT}/val/safety_preferences.jsonl"
GPUS=(2 3)
BETAS=(0.05 0.20)
NAMES=(beta005 beta020)

for path in \
  "${MODEL}/adapter_model.safetensors" \
  "${MODEL}/adapter_config.json" \
  "${TRAIN_FILE}" \
  "${EVAL_FILE}"; do
  [[ -s "${path}" ]] || { echo "Required input is missing: ${path}" >&2; exit 2; }
done

for gpu in "${GPUS[@]}"; do
  pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits)"
  [[ -z "${pids//[[:space:]]/}" ]] || {
    echo "GPU ${gpu} is occupied: ${pids}" >&2
    exit 3
  }
done

PYTHONPATH="${PROJECT_ROOT}/src:${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${PYTHON}" -c "import datasets, peft, trl, transformers"

for index in "${!GPUS[@]}"; do
  gpu="${GPUS[$index]}"
  beta="${BETAS[$index]}"
  name="${NAMES[$index]}"
  run_dir="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_safety_dpo_${name}_seed20260821_v1"
  [[ ! -e "${run_dir}" ]] || {
    echo "Refusing existing run directory: ${run_dir}" >&2
    exit 4
  }
  mkdir -p "${run_dir}/control"
  cat >"${run_dir}/launch.txt" <<EOF
protocol=muno21-safety-dpo-beta-ablation-v1
physical_gpu=${gpu}
model=${MODEL}
train_file=${TRAIN_FILE}
eval_file=${EVAL_FILE}
beta=${beta}
seed=20260821
test_assets_read=false
EOF
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PROJECT_ROOT}/src:${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON}" "${PROJECT_ROOT}/scripts/train_agent_dpo.py" \
      "${MODEL}" "${TRAIN_FILE}" "${run_dir}" \
      --eval-jsonl "${EVAL_FILE}" \
      --epochs 1 --learning-rate 1e-5 --batch-size 1 \
      --gradient-accumulation 32 --max-length 2048 --beta "${beta}" \
      --logging-steps 5 --eval-steps 25 --save-steps 25 --seed 20260821 \
      >"${run_dir}/train.log" 2>&1 < /dev/null &
  pid="$!"
  echo "${pid}" >"${run_dir}/train.pid"
  echo "${name}: gpu=${gpu} pid=${pid}"
done
