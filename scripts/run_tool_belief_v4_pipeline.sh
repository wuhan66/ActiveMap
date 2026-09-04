#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/wh/projects/activemap-v1}"
source "${PROJECT_ROOT}/scripts/server_hdpi_env.sh"
export ACTIVEMAP_LAUNCHER_NAME="run_tool_belief_v4_pipeline.sh"
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/scripts/acquire_training_slot.sh"
DATA_ROOT="${ACTIVEMAP_STORAGE_ROOT}"
PYTHON="${ACTIVEMAP_PYTHON:-${ACTIVEMAP_GIS_ENV}/bin/python}"
PHYSICAL_GPU="${PHYSICAL_GPU:-${ACTIVEMAP_GPU_IDS%%,*}}"
SEED="${SEED:-20260821}"
RUN_NAME="${RUN_NAME:-tool_belief_anchored_v4_seed${SEED}}"
FALSE_EDIT_WEIGHT="${FALSE_EDIT_WEIGHT:-2.0}"
SEQUENCE_ROOT="${DATA_ROOT}/processed/muno21_v2/agent/tool_belief_sequence_v1"
RUN_ROOT="${DATA_ROOT}/runs/agent/${RUN_NAME}"
EVAL_ROOT="${DATA_ROOT}/runs/agent/${RUN_NAME}_sequence_eval"
TRAIN_EVAL_ROOT="${DATA_ROOT}/runs/agent/${RUN_NAME}_sequence_eval_train"
REPORT_ROOT="${DATA_ROOT}/runs/agent/${RUN_NAME}_report"
LOG_ROOT="${DATA_ROOT}/logs"
TRAIN_LOG="${LOG_ROOT}/train_${RUN_NAME}.log"
EVAL_LOG="${LOG_ROOT}/eval_${RUN_NAME}.log"

cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

if [[ ! "${PHYSICAL_GPU}" =~ ^(0|1)$ ]]; then
  echo "refusing GPU ${PHYSICAL_GPU}: ActiveMap is restricted to physical GPUs 0 and 1" >&2
  exit 2
fi
PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON}" \
  scripts/assert_training_ready.py \
  configs/experiments/paper_registry.yaml "${DATA_ROOT}"

GPU_UUID="$(nvidia-smi --id="${PHYSICAL_GPU}" --query-gpu=uuid --format=csv,noheader)"
if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
  | grep -Fxq "${GPU_UUID}"; then
  echo "refusing to start: physical GPU ${PHYSICAL_GPU} has a compute process" >&2
  exit 21
fi

for protected in \
  "${RUN_ROOT}/history.jsonl" \
  "${RUN_ROOT}/best.pt" \
  "${RUN_ROOT}/summary.json" \
  "${EVAL_ROOT}/summary.json" \
  "${TRAIN_EVAL_ROOT}/summary.json"; do
  if [[ -e "${protected}" ]]; then
    echo "refusing to overwrite existing artifact: ${protected}" >&2
    exit 22
  fi
done

echo "run=${RUN_NAME} physical_gpu=${PHYSICAL_GPU} seed=${SEED} false_edit_weight=${FALSE_EDIT_WEIGHT}"
echo "started_at=$(date --iso-8601=seconds)"

CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}" PYTHONPATH=src:. "${PYTHON}" -u \
  scripts/train_tool_belief_recurrent.py \
  "${SEQUENCE_ROOT}/train.jsonl" \
  "${SEQUENCE_ROOT}/val.jsonl" \
  "${RUN_ROOT}" \
  --device cuda:0 \
  --fusion-mode paired_anchored \
  --false-edit-weight "${FALSE_EDIT_WEIGHT}" \
  --seed "${SEED}" 2>&1 | tee "${TRAIN_LOG}"

CHECKPOINT="${RUN_ROOT}/best.pt"
if [[ -f "${RUN_ROOT}/best_safety.pt" ]]; then
  CHECKPOINT="${RUN_ROOT}/best_safety.pt"
fi
echo "evaluation_checkpoint=${CHECKPOINT}"

PYTHONPATH=src:. "${PYTHON}" scripts/evaluate_tool_belief_sequences.py \
  "${SEQUENCE_ROOT}/val.jsonl" \
  "${CHECKPOINT}" \
  "${EVAL_ROOT}" \
  --device cpu 2>&1 | tee "${EVAL_LOG}"

PYTHONPATH=src:. "${PYTHON}" scripts/evaluate_tool_belief_sequences.py \
  "${SEQUENCE_ROOT}/train.jsonl" \
  "${CHECKPOINT}" \
  "${TRAIN_EVAL_ROOT}" \
  --device cpu 2>&1 | tee "${LOG_ROOT}/eval_${RUN_NAME}_train.log"

PYTHONPATH=src:. "${PYTHON}" scripts/analyze_tool_belief_stopping.py \
  "${EVAL_ROOT}/summary.json" \
  "${EVAL_ROOT}/details.jsonl" \
  "${REPORT_ROOT}/stopping_opportunity.json" \
  --plot "${REPORT_ROOT}/stopping_opportunity.png"

PYTHONPATH=src:. "${PYTHON}" scripts/render_tool_belief_report.py \
  "${RUN_ROOT}/history.jsonl" \
  "${EVAL_ROOT}/summary.json" \
  "${REPORT_ROOT}"

echo "completed_at=$(date --iso-8601=seconds)"
echo "run_root=${RUN_ROOT}"
echo "eval_root=${EVAL_ROOT}"
echo "train_eval_root=${TRAIN_EVAL_ROOT}"
echo "report_root=${REPORT_ROOT}"
