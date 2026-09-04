#!/usr/bin/env bash
set -euo pipefail

# Formal v4 SELECT replication. All choices below were frozen after the
# separate, same-seed reliability-weight pilot screen. The script evaluates
# only the frozen validation split and aggregates after all three seeds finish.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
TRAIN="${TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/train/selector_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v4_crossfit_supported_select/val/selector_sft.jsonl}"
LOG_ROOT="${LOG_ROOT:-${STORE}/logs}"
if [[ -n "${GPU_LIST:-}" ]]; then
  read -r -a GPUS <<<"${GPU_LIST}"
else
  GPUS=(1 2 3)
fi
SEEDS=(20260881 20260882 20260883)
PREFIX="muno21_qwen3vl_crossfit_supported_weight075_formal"
AGGREGATE="${STORE}/runs/agent/${PREFIX}_3seed_v2"

[[ ${#GPUS[@]} -ge ${#SEEDS[@]} ]] || {
  echo "Need at least ${#SEEDS[@]} GPUs, got ${#GPUS[@]}" >&2
  exit 2
}
cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${TRAIN}" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 3; }
done
[[ ! -e "${AGGREGATE}" ]] || { echo "Refusing existing aggregate: ${AGGREGATE}" >&2; exit 4; }
mkdir -p "${LOG_ROOT}"

pids=()
for index in "${!SEEDS[@]}"; do
  seed="${SEEDS[$index]}"
  gpu="${GPUS[$index]}"
  name="${PREFIX}_seed${seed}"
  run="${STORE}/runs/agent/${name}"
  evaluation="${STORE}/runs/agent/${name}_eval"
  log="${LOG_ROOT}/${name}.log"
  [[ ! -e "${run}" && ! -e "${evaluation}" ]] || {
    echo "Refusing existing artifact for ${name}" >&2
    exit 5
  }
  (
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 \
      "${PY}" scripts/train_semantic_vlm_sft.py \
        "${MODEL}" "${TRAIN}" "${run}" --eval-jsonl "${VAL}" \
        --epochs 2.9 --learning-rate 0.0001 --batch-size 1 --gradient-accumulation 16 \
        --max-length 2048 --lora-rank 16 --lora-alpha 32 --seed "${seed}" \
        --acquire-sampling-target 0.15 --record-weight-key crossfit_reliability_weight \
        --logging-steps 5 --eval-steps 50 --save-steps 50 --save-total-limit 4 \
        --early-stopping-patience 3 --dataloader-num-workers 2 --dataloader-persistent-workers \
        >"${log}" 2>&1
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
      TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
      "${PY}" scripts/evaluate_sequential_selector.py \
        "${MODEL}" "${run}/final" "${VAL}" "${evaluation}" --device cuda:0 \
        --expected-records 414 --bootstrap-repetitions 10000 --seed "${seed}" \
        >>"${log}" 2>&1
  ) &
  pid="$!"
  pids+=("${pid}")
  echo "started ${name} on physical GPU ${gpu}: pid=${pid}"
done

status=0
for pid in "${pids[@]}"; do
  wait "${pid}" || status=1
done
[[ "${status}" == 0 ]] || exit "${status}"

PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" \
  scripts/aggregate_sequential_selector_replicates.py "${AGGREGATE}" \
  --run "seed20260881=${STORE}/runs/agent/${PREFIX}_seed20260881_eval" \
  --run "seed20260882=${STORE}/runs/agent/${PREFIX}_seed20260882_eval" \
  --run "seed20260883=${STORE}/runs/agent/${PREFIX}_seed20260883_eval" \
  --bootstrap-repetitions 10000 --seed 20260881 \
  >"${LOG_ROOT}/${PREFIX}_3seed_v2.aggregate.log" 2>&1
