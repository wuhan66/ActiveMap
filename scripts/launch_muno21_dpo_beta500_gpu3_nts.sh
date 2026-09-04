#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
PYTHON="${ACTIVEMAP_AGENT_PYTHON:-/home/wh/venvs/activemap/bin/python}"
OVERLAY="${ACTIVEMAP_AGENT_SITE_PACKAGES:-${STORAGE_ROOT}/envs/agent_peft_overlay}"
MODEL="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_sft_v3_anonymized_seed20260821/checkpoints/checkpoint-200"
DATA_ROOT="${STORAGE_ROOT}/processed/muno21_v2/agent/agent_data_v6_anonymized"
TRAIN_FILE="${DATA_ROOT}/train/safety_preferences.jsonl"
EVAL_FILE="${DATA_ROOT}/val/safety_preferences.jsonl"
RUN_DIR="${STORAGE_ROOT}/runs/agent/muno21_qwen3_4b_safety_dpo_beta500_seed20260821_v1"
GPU=3

for path in "${MODEL}/adapter_model.safetensors" "${MODEL}/adapter_config.json" \
  "$TRAIN_FILE" "$EVAL_FILE"; do
  [[ -s "$path" ]] || { echo "Required input is missing: $path" >&2; exit 2; }
done
[[ ! -e "$RUN_DIR" ]] || { echo "Refusing existing run: $RUN_DIR" >&2; exit 3; }
mkdir -p "${RUN_DIR}/control"
cat >"${RUN_DIR}/launch.txt" <<EOF
protocol=muno21-safety-dpo-high-beta-ablation-v1
physical_gpu=${GPU}
beta=5.0
seed=20260821
test_assets_read=false
EOF

cd "$PROJECT_ROOT"
nohup env CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 \
  PYTHONPATH="${PROJECT_ROOT}/src:${OVERLAY}${PYTHONPATH:+:${PYTHONPATH}}" \
  "$PYTHON" scripts/train_agent_dpo.py \
    "$MODEL" "$TRAIN_FILE" "$RUN_DIR" --eval-jsonl "$EVAL_FILE" \
    --epochs 1 --learning-rate 1e-5 --batch-size 1 --gradient-accumulation 32 \
    --max-length 2048 --beta 5.0 --logging-steps 5 --eval-steps 25 \
    --save-steps 25 --seed 20260821 >"${RUN_DIR}/train.log" 2>&1 < /dev/null &
echo "$!" >"${RUN_DIR}/train.pid"
echo "started=beta500 gpu=${GPU} pid=$!"
