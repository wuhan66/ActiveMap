#!/usr/bin/env bash
set -euo pipefail

cd /home/wh/projects/activemap-v1-joint-debug
export PYTHONPATH=src:.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false

PY=/home/wh/venvs/activemap/bin/python
ROOT=/mnt/mydisk/wh/ActiveMap
MODEL=${ROOT}/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821/checkpoints/checkpoint-200
TRAIN=${ROOT}/processed/muno21_v2/agent/agent_data_v6_anonymized/train/safety_preferences.jsonl
VAL=${ROOT}/processed/muno21_v2/agent/agent_data_v6_anonymized/val/safety_preferences.jsonl

launch_one() {
  local gpu=$1
  local seed=$2
  local out=${ROOT}/runs/agent/muno21_qwen3_4b_safety_dpo_beta020_seed${seed}_v1
  mkdir -p "${out}"
  cat >"${out}/launch.txt" <<EOF
protocol=muno21-safety-dpo-beta-ablation-v1
physical_gpu=${gpu}
model=${MODEL}
train_file=${TRAIN}
eval_file=${VAL}
beta=0.20
seed=${seed}
test_assets_read=false
EOF
  CUDA_VISIBLE_DEVICES=${gpu} "${PY}" scripts/train_agent_dpo.py \
    "${MODEL}" "${TRAIN}" "${out}" \
    --eval-jsonl "${VAL}" \
    --epochs 1 \
    --learning-rate 1e-5 \
    --batch-size 1 \
    --gradient-accumulation 32 \
    --max-length 2048 \
    --beta 0.20 \
    --logging-steps 5 \
    --eval-steps 25 \
    --save-steps 25 \
    --seed "${seed}" >"${out}/train.log" 2>&1 &
  echo "$! GPU${gpu} beta0.20 seed${seed}"
}

launch_one 1 20260822
launch_one 2 20260823
wait
