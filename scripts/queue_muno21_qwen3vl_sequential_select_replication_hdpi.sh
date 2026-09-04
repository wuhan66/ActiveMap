#!/usr/bin/env bash
set -euo pipefail

# Two preregistered independent replicas of the only promising SELECT policy.
# The original 25% ACQUIRE model had positive utility but its single-model task
# bootstrap CI narrowly crossed zero.  These replicas establish seed stability;
# they do not launch terminal SFT, joint SFT, RL, or frozen-test evaluation.

PROJECT="${PROJECT:-/home/wh/projects/activemap-v1}"
STORE="${STORE:-/home/wh/ActiveMap}"
PY="${PY:-${STORE}/envs/activemap-agent/bin/python}"
MODEL="${MODEL:-/home/wh/hf_models/Qwen3-VL-4B-Instruct}"
TRAIN="${TRAIN:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/train/selector_balanced_sft.jsonl}"
VAL="${VAL:-${STORE}/processed/muno21_v2/agent/sequential_controller_qwen_v1/val/selector_sft.jsonl}"
OUTPUT="${OUTPUT:-${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_primary_3seed_v1}"
POLL_SECONDS="${POLL_SECONDS:-90}"

cd "${PROJECT}"
for path in "${PY}" "${MODEL}/config.json" "${TRAIN}" "${VAL}"; do
  [[ -s "${path}" ]] || { echo "Missing input: ${path}" >&2; exit 2; }
done
[[ ! -e "${OUTPUT}" ]] || { echo "Refusing existing output: ${OUTPUT}" >&2; exit 3; }
# The aggregation command owns OUTPUT and refuses an existing directory. Keep
# its launcher log beside the result directory rather than pre-creating it.
mkdir -p "${STORE}/logs"

run_seed() {
  local gpu="$1" seed="$2"
  local name="muno21_qwen3vl4b_sequential_select_primary_v1_seed${seed}"
  local run="${STORE}/runs/agent/${name}"
  local eval="${STORE}/runs/agent/${name}_eval"
  [[ ! -e "${run}" && ! -e "${eval}" ]] || { echo "Existing artifact for ${name}" >&2; return 4; }
  echo "[$(date '+%F %T')] training ${name} on GPU${gpu}"
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
  echo "[$(date '+%F %T')] evaluating ${name} on GPU${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH="${PROJECT}/src:${PROJECT}" \
    TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    "${PY}" scripts/evaluate_sequential_selector.py \
      "${MODEL}" "${run}/final" "${VAL}" "${eval}" --device cuda:0 \
      --expected-records 414 --bootstrap-repetitions 2000 --seed "${seed}" \
      >"${STORE}/logs/${name}.eval.log" 2>&1
  echo "[$(date '+%F %T')] completed ${name}"
}

run_seed 1 20260863 &
seed1_pid=$!
run_seed 2 20260864 &
seed2_pid=$!
wait "${seed1_pid}"
wait "${seed2_pid}"

echo "[$(date '+%F %T')] aggregating three fixed primary seeds"
PYTHONPATH="${PROJECT}/src:${PROJECT}" "${PY}" \
  scripts/aggregate_sequential_selector_replicates.py "${OUTPUT}" \
  --run "seed20260862=${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_v1_seed20260862_eval" \
  --run "seed20260863=${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_primary_v1_seed20260863_eval" \
  --run "seed20260864=${STORE}/runs/agent/muno21_qwen3vl4b_sequential_select_primary_v1_seed20260864_eval" \
  --bootstrap-repetitions 10000 --seed 20260865 \
  >"${OUTPUT}.aggregate.log" 2>&1
echo "[$(date '+%F %T')] replication complete: ${OUTPUT}"
