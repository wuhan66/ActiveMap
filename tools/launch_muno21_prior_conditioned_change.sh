#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1-joint-debug}"
STORAGE_ROOT="${STORAGE_ROOT:-/mnt/mydisk/wh/ActiveMap}"
GPU="${SAM_ROAD_GPU:-3}"
ENV_PREFIX="${SAM_ROAD_ENV:-/home/wh/yes/envs/activemap-sam-road}"
RUN_NAME="${RUN_NAME:-muno21_prior_conditioned_change_seed20260716_v2}"
RUN_DIR="${RUN_DIR:-${STORAGE_ROOT}/runs/semantic/${RUN_NAME}}"
LOG_PATH="${LOG_PATH:-${STORAGE_ROOT}/logs/${RUN_NAME}.log}"
SMOKE="${SMOKE:-0}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
PATIENCE="${PATIENCE:-6}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
SEED="${SEED:-20260716}"

DATASET="${STORAGE_ROOT}/processed/muno21_v2/rsprompter_road_v2_rle"
UPDATER_MANIFEST="${STORAGE_ROOT}/processed/muno21_v2/updater/updater_samples.jsonl"
SAM_REPO="${STORAGE_ROOT}/external/sam_road"
SAM_CONFIG="${SAM_REPO}/config/toponet_vitb_256_spacenet.yaml"
SOURCE_CHECKPOINT="${STORAGE_ROOT}/runs/semantic/muno21_sam_road_head_seed20260716_v1/checkpoints/best_full.ckpt"
SAM_CHECKPOINT="${STORAGE_ROOT}/models/sam_road/sam_vit_b_01ec64.pth"
CODE_RECEIPT="${CODE_RECEIPT:-$(ls -1t "${STORAGE_ROOT}/code-sync/receipts/"*.txt 2>/dev/null | head -n 1)}"
DATA_INVENTORY="${DATA_INVENTORY:-${STORAGE_ROOT}/data-versions/current.tsv}"

for path in \
  "${ENV_PREFIX}/bin/python" "$DATASET" "$UPDATER_MANIFEST" "$SAM_REPO" \
  "$SAM_CONFIG" "$SOURCE_CHECKPOINT" "$SAM_CHECKPOINT" "$CODE_RECEIPT" \
  "$DATA_INVENTORY"; do
  test -e "$path" || { echo "missing prerequisite: $path" >&2; exit 2; }
done
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run: $RUN_DIR" >&2
  exit 2
fi

IFS=',' read -r utilization memory_used < <(
  nvidia-smi --id="$GPU" \
    --query-gpu=utilization.gpu,memory.used \
    --format=csv,noheader,nounits | tr -d ' '
)
if (( utilization > 10 || memory_used > 1024 )); then
  echo "GPU ${GPU} is not idle: utilization=${utilization}%, memory=${memory_used} MiB" >&2
  exit 3
fi

mkdir -p "$(dirname "$LOG_PATH")"
cd "$PROJECT_ROOT"
export ACTIVEMAP_DISABLE_FROZEN_TEST=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="$GPU"

smoke_args=()
if [[ "$SMOKE" == "1" ]]; then
  smoke_args+=(--smoke --workers 0)
fi
operation_args=()
if [[ -n "${OPERATION_ONLY_FROM:-}" ]]; then
  test -f "$OPERATION_ONLY_FROM" || {
    echo "missing operation source checkpoint: $OPERATION_ONLY_FROM" >&2
    exit 2
  }
  operation_args+=(--operation-only-from "$OPERATION_ONLY_FROM")
fi
if [[ -n "${MINIMUM_CALIBRATION_MACRO_F1:-}" ]]; then
  operation_args+=(
    --minimum-calibration-macro-f1 "$MINIMUM_CALIBRATION_MACRO_F1"
  )
fi
if [[ "${POSITIVE_ONLY_CHANGE_DICE:-0}" == "1" ]]; then
  operation_args+=(--positive-only-change-dice)
fi

exec "${ENV_PREFIX}/bin/python" scripts/train_muno21_prior_conditioned_change.py \
  "$DATASET" "$UPDATER_MANIFEST" "$SAM_REPO" "$SAM_CONFIG" \
  "$SOURCE_CHECKPOINT" "$SAM_CHECKPOINT" "$RUN_DIR" \
  --calibration-aoi atlanta \
  --device cuda:0 \
  --seed "$SEED" \
  --batch-size 8 \
  --workers 4 \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --learning-rate "$LEARNING_RATE" \
  --decoder-learning-rate 3e-5 \
  --weight-decay 1e-4 \
  --max-positive-weight 60.0 \
  --max-false-edit 0.10 \
  --change-parameterization "${CHANGE_PARAMETERIZATION:-independent}" \
  --operation-head "${OPERATION_HEAD:-global_stats}" \
  --code-receipt "$CODE_RECEIPT" \
  --data-inventory "$DATA_INVENTORY" \
  "${operation_args[@]}" \
  "${smoke_args[@]}" \
  2>&1 | tee "$LOG_PATH"
