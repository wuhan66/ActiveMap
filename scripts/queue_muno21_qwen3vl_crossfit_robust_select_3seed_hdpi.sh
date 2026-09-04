#!/usr/bin/env bash
set -euo pipefail

# Fixed A1 repair: every train label is positive only when both the on-policy
# branch and a task-held-out cross-fit branch realize positive utility. The
# validation split is separate and never appears in TRAIN.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
TRAIN="${TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v3_crossfit_robust_split/train/selector_balanced_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v3_crossfit_robust_split/val/selector_sft.jsonl}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_qwen3vl_crossfit_robust_select_3seed_v1}"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${TRAIN}" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing aggregate output: ${OUTPUT}" >&2; exit 3; }
mkdir -p "${STORE}/logs"

run_seed() {
  local gpu="$1" seed="$2"
  local name="muno21_qwen3vl_crossfit_robust_select_v1_seed${seed}"
  local run="${STORE}/runs/agent/${name}"
  local evaluation="${STORE}/runs/agent/${name}_eval"
  [[ ! -e "${run}" && ! -e "${evaluation}" ]] || { echo "Existing artifact: ${name}" >&2; return 4; }
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
    "${PY}" scripts/train_semantic_vlm_sft.py \
      "${MODEL}" "${TRAIN}" "${run}" --eval-jsonl "${VAL}" \
      --epochs 2 --learning-rate 0.0001 --batch-size 1 --gradient-accumulation 16 \
      --max-length 2048 --lora-rank 16 --lora-alpha 32 --seed "${seed}" \
      --acquire-sampling-target 0.25 --logging-steps 5 --eval-steps 50 \
      --save-steps 50 --save-total-limit 4 --early-stopping-patience 3 \
      --dataloader-num-workers 2 --dataloader-persistent-workers \
      >"${STORE}/logs/${name}.log" 2>&1
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "${PY}" scripts/evaluate_sequential_selector.py \
      "${MODEL}" "${run}/final" "${VAL}" "${evaluation}" --device cuda:0 \
      --expected-records 414 --bootstrap-repetitions 10000 --seed "${seed}" \
      >"${STORE}/logs/${name}.eval.log" 2>&1
}

run_seed 1 20260871 & pid1=$!
run_seed 2 20260872 & pid2=$!
run_seed 3 20260873 & pid3=$!
wait "${pid1}"
wait "${pid2}"
wait "${pid3}"

PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" \
  scripts/aggregate_sequential_selector_replicates.py "${OUTPUT}" \
  --run "seed20260871=${STORE}/runs/agent/muno21_qwen3vl_crossfit_robust_select_v1_seed20260871_eval" \
  --run "seed20260872=${STORE}/runs/agent/muno21_qwen3vl_crossfit_robust_select_v1_seed20260872_eval" \
  --run "seed20260873=${STORE}/runs/agent/muno21_qwen3vl_crossfit_robust_select_v1_seed20260873_eval" \
  --bootstrap-repetitions 10000 --seed 20260874 \
  >"${OUTPUT}.aggregate.log" 2>&1
