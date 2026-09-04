#!/usr/bin/env bash
set -euo pipefail

# Exploratory pilot only. The primary on-policy utility remains the target;
# cross-fit agreement is supplied as train-only loss reliability metadata.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
GPU="${GPU:-4}"
SEED="${SEED:-20260875}"
TRAIN="${TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/train/selector_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/val/selector_sft.jsonl}"
RELIABILITY_WEIGHT_KEY="${RELIABILITY_WEIGHT_KEY:-crossfit_reliability_weight}"
NAME="${NAME:-muno21_qwen3vl_crossfit_supported_select_pilot_seed${SEED}}"
RUN="${STORE}/runs/agent/${NAME}"
EVALUATION="${STORE}/runs/agent/${NAME}_eval"
LOG="${STORE}/logs/${NAME}.log"
EVAL_LOG="${STORE}/logs/${NAME}.eval.log"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${TRAIN}" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${RUN}" && ! -e "${EVALUATION}" ]] || {
  echo "Refusing existing artifact: ${NAME}" >&2
  exit 3
}
mkdir -p "${STORE}/logs"
record_weight_args=()
if [[ "${RELIABILITY_WEIGHT_KEY}" != "NONE" && -n "${RELIABILITY_WEIGHT_KEY}" ]]; then
  record_weight_args=(--record-weight-key "${RELIABILITY_WEIGHT_KEY}")
fi
train_args=(
  scripts/train_semantic_vlm_sft.py
  "${MODEL}" "${TRAIN}" "${RUN}" --eval-jsonl "${VAL}"
  --epochs 2.9 --learning-rate 0.0001 --batch-size 1 --gradient-accumulation 16
  --max-length 2048 --lora-rank 16 --lora-alpha 32 --seed "${SEED}"
  --acquire-sampling-target 0.15
  --logging-steps 5 --eval-steps 50 --save-steps 50 --save-total-limit 4
  --early-stopping-patience 3 --dataloader-num-workers 2 --dataloader-persistent-workers
)
train_args+=("${record_weight_args[@]}")

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  "${PY}" "${train_args[@]}" >"${LOG}" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "${PY}" scripts/evaluate_sequential_selector.py \
    "${MODEL}" "${RUN}/final" "${VAL}" "${EVALUATION}" --device cuda:0 \
    --expected-records 414 --bootstrap-repetitions 10000 --seed "${SEED}" \
    >"${EVAL_LOG}" 2>&1
