#!/usr/bin/env bash
set -euo pipefail

# One additional fixed SELECT replica. It increases the precision of the
# already preregistered primary configuration; it is not a hyperparameter sweep.
PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
TRAIN="${TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/train/selector_balanced_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/val/selector_sft.jsonl}"
SEED="${SEED:-20260865}"
GPU="${GPU:-3}"
RUN_NAME="muno21_qwen3vl4b_sequential_select_primary_v1_seed${SEED}"
RUN="${STORE}/runs/agent/${RUN_NAME}"
EVAL="${STORE}/runs/agent/${RUN_NAME}_eval"
OUTPUT="${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_primary_4seed_v1"
LOG="${STORE}/logs/${RUN_NAME}.log"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${TRAIN}" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${RUN}" && ! -e "${EVAL}" && ! -e "${OUTPUT}" ]] || {
  echo "Existing artifact blocks fourth-seed queue" >&2; exit 3;
}

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
  "${PY}" scripts/train_semantic_vlm_sft.py \
    "${MODEL}" "${TRAIN}" "${RUN}" --eval-jsonl "${VAL}" \
    --epochs 2 --learning-rate 0.0001 --batch-size 1 --gradient-accumulation 16 \
    --max-length 2048 --lora-rank 16 --lora-alpha 32 --seed "${SEED}" \
    --acquire-sampling-target 0.25 --logging-steps 5 --eval-steps 50 \
    --save-steps 50 --save-total-limit 4 --early-stopping-patience 3 \
    --dataloader-num-workers 2 --dataloader-persistent-workers \
    >"${LOG}" 2>&1

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
  TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "${PY}" scripts/evaluate_sequential_selector.py \
    "${MODEL}" "${RUN}/final" "${VAL}" "${EVAL}" --device cuda:0 \
    --expected-records 414 --bootstrap-repetitions 2000 --seed "${SEED}" \
    >"${STORE}/logs/${RUN_NAME}.eval.log" 2>&1

for prior in 20260863 20260864; do
  path="${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_primary_v1_seed${prior}_eval/summary.json"
  until [[ -s "${path}" ]]; do sleep 90; done
done
PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" \
  scripts/aggregate_sequential_selector_replicates.py "${OUTPUT}" \
  --run "seed20260862=${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_v1_seed20260862_eval" \
  --run "seed20260863=${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_primary_v1_seed20260863_eval" \
  --run "seed20260864=${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_primary_v1_seed20260864_eval" \
  --run "seed20260865=${EVAL}" --bootstrap-repetitions 10000 --seed 20260866 \
  >"${OUTPUT}.log" 2>&1
